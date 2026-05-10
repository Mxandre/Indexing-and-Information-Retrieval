"""Analyseur de requêtes avec correction orthographique par distance de Levenshtein.

Charge un lexique TSV (mot → lemme), tokenise une requête, valide chaque token
contre le lexique et propose des corrections basées sur le préfixe commun et la
distance de Levenshtein pour les tokens hors-lexique.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MINI_LEXICON = BASE_DIR / "mini_lexicon.tsv"
DEFAULT_FULL_LEXICON = BASE_DIR.parent / "TD3" / "mot_lemma_list.txt"

TOKEN_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d+(?:[.,]\d+)?"
    r"|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*"
)

DATE_PATTERN = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})$")
NUMBER_PATTERN = re.compile(r"^\d+(?:[.,]\d+)?$")


def strip_accents(text: str) -> str:
    """Supprime les accents d'une chaîne de caractères."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize(text: str) -> str:
    """Normalise un token : minuscules, suppression des accents, apostrophes typographiques.

    Args:
        text (str): Token brut.

    Returns:
        str: Token normalisé.
    """
    text = text.strip().lower().replace("’", "'")
    return strip_accents(text)


def tokenize(text: str) -> list[str]:
    """Tokenise une chaîne de caractères selon le patron TOKEN_PATTERN.

    Args:
        text (str): Texte brut.

    Returns:
        list[str]: Liste de tokens extraits.
    """
    return TOKEN_PATTERN.findall(text.replace("’", "'"))


def common_prefix_length(left: str, right: str) -> int:
    """Calcule la longueur du préfixe commun entre deux chaînes.

    Args:
        left (str): Première chaîne.
        right (str): Deuxième chaîne.

    Returns:
        int: Nombre de caractères consécutifs identiques depuis le début.
    """
    limit = min(len(left), len(right))
    size = 0
    while size < limit and left[size] == right[size]:
        size += 1
    return size


def levenshtein_distance(left: str, right: str) -> int:
    """Calcule la distance de Levenshtein entre deux chaînes.

    Args:
        left (str): Chaîne source.
        right (str): Chaîne cible.

    Returns:
        int: Nombre minimal d'insertions, suppressions ou substitutions.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_left != char_right)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def is_specific_entity(token: str) -> bool:
    """Vérifie si un token est une date ou un nombre (entité à conserver telle quelle).

    Args:
        token (str): Token à tester.

    Returns:
        bool: ``True`` si le token est une date ou un nombre.
    """
    return bool(DATE_PATTERN.match(token) or NUMBER_PATTERN.match(token))


def safe_spacy_tokenize_and_lemmatize(text: str) -> list[tuple[str, str]]:
    """Tokenise et lemmatise une chaîne avec spaCy si disponible.

    Args:
        text (str): Texte à analyser.

    Returns:
        list[tuple[str, str]]: Paires ``(forme_originale, lemme_minuscule)``,
        ou liste vide si spaCy est indisponible.
    """
    try:
        import spacy 

        nlp = spacy.load("fr_core_news_sm")
    except Exception:
        return []

    return [
        (token.text, token.lemma_.lower())
        for token in nlp(text)
        if not token.is_space and not token.is_punct
    ]


@dataclass(frozen=True)
class Candidate:
    """Représente un candidat de correction pour un token hors-lexique."""

    word: str
    lemma: str
    prefix_len: int
    prefix_ratio: float
    distance: int


class Lexicon:
    """Lexique mot → lemme chargé depuis un fichier TSV."""

    def __init__(self, entries: dict[str, str], source: Path) -> None:
        """Initialise le lexique.

        Args:
            entries (dict[str, str]): Dictionnaire ``{mot_normalisé: lemme}``.
            source (Path): Chemin du fichier source.
        """
        self.entries = entries
        self.source = source
        self.words = sorted(entries)

    @classmethod
    def from_tsv(cls, path: Path) -> "Lexicon":
        """Charge un lexique depuis un fichier TSV ``mot \tab lemme``.

        En cas de lemmes multiples pour un même mot, conserve le plus fréquent.

        Args:
            path (Path): Chemin du fichier TSV.

        Returns:
            Lexicon: Instance chargée.
        """
        lemma_votes: dict[str, Counter[str]] = defaultdict(Counter)
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or "\t" not in line:
                    continue
                word, lemma = line.split("\t", maxsplit=1)
                lemma_votes[normalize(word)][normalize(lemma)] += 1

        entries = {
            word: votes.most_common(1)[0][0]
            for word, votes in lemma_votes.items()
            if word
        }
        return cls(entries=entries, source=path)

    def contains(self, token: str) -> bool:
        """Indique si le token normalisé est présent dans le lexique."""
        return normalize(token) in self.entries

    def lemma_for(self, token: str) -> str | None:
        """Retourne le lemme d'un token s'il est présent dans le lexique.

        Args:
            token (str): Token à chercher.

        Returns:
            str | None: Lemme correspondant, ou ``None`` si absent.
        """
        return self.entries.get(normalize(token))

    def generate_candidates(
        self,
        token: str,
        seuil_min: int,
        seuil_max: int,
        seuil_proximite: float,
    ) -> list[Candidate]:
        """Génère les candidats de correction pour un token hors-lexique.

        Filtre les mots du lexique selon la longueur du préfixe commun,
        la différence de longueur et la proximité préfixale, puis trie les
        résultats par distance de Levenshtein croissante.

        Args:
            token (str): Token à corriger.
            seuil_min (int): Longueur minimale du préfixe commun.
            seuil_max (int): Différence maximale de longueur tolérée.
            seuil_proximite (float): Ratio minimal de proximité préfixale.

        Returns:
            list[Candidate]: Candidats triés par ``(distance, -prefix_len, mot)``.
        """
        normalized_token = normalize(token)
        token_length = len(normalized_token)
        if not normalized_token:
            return []

        candidates: list[Candidate] = []
        for word in self.words:
            prefix_len = common_prefix_length(normalized_token, word)
            prefix_ratio = prefix_len / max(1, min(token_length, len(word)))
            if prefix_len < seuil_min:
                continue
            if abs(len(word) - token_length) > seuil_max:
                continue
            if prefix_ratio < seuil_proximite:
                continue
            candidates.append(
                Candidate(
                    word=word,
                    lemma=self.entries[word],
                    prefix_len=prefix_len,
                    prefix_ratio=prefix_ratio,
                    distance=levenshtein_distance(normalized_token, word),
                )
            )

        return sorted(candidates, key=lambda item: (item.distance, -item.prefix_len, item.word))


def analyze_query(
    query: str,
    lexicon: Lexicon,
    seuil_min: int,
    seuil_max: int,
    seuil_proximite: float,
) -> list[dict[str, object]]:
    """Analyse une requête en validant et corrigeant chaque token via le lexique.

    Utilise spaCy si disponible pour la tokenisation et la lemmatisation, sinon
    recourt au patron TOKEN_PATTERN. Pour chaque token : entité reconnue → conservé,
    présent dans le lexique → validé, absent → candidats de correction générés.

    Args:
        query (str): Requête saisie par l'utilisateur.
        lexicon (Lexicon): Lexique de référence.
        seuil_min (int): Longueur minimale du préfixe commun.
        seuil_max (int): Différence maximale de longueur tolérée.
        seuil_proximite (float): Ratio minimal de proximité préfixale.

    Returns:
        list[dict[str, object]]: Un dictionnaire par token avec les clés
        ``token``, ``status``, ``correction``, ``lemma``, ``candidates``.
    """
    spacy_tokens = safe_spacy_tokenize_and_lemmatize(query)
    tokens = [token for token, _lemma in spacy_tokens] if spacy_tokens else tokenize(query)

    results: list[dict[str, object]] = []
    for token in tokens:
        normalized = normalize(token)
        if not normalized:
            continue

        if is_specific_entity(token):
            results.append({
                "token": token,
                "status": "entite",
                "correction": token,
                "lemma": token,
                "candidates": [],
            })
            continue

        direct_lemma = lexicon.lemma_for(token)
        if direct_lemma is not None:
            results.append({
                "token": token,
                "status": "valide",
                "correction": normalized,
                "lemma": direct_lemma,
                "candidates": [],
            })
            continue

        candidates = lexicon.generate_candidates(
            token=token,
            seuil_min=seuil_min,
            seuil_max=seuil_max,
            seuil_proximite=seuil_proximite,
        )

        if not candidates:
            results.append({
                "token": token,
                "status": "introuvable",
                "correction": None,
                "lemma": None,
                "candidates": [],
            })
            continue

        if len(candidates) == 1:
            chosen = candidates[0]
            status = "corrige-candidat-unique"
        else:
            best_distance = candidates[0].distance
            best = [c for c in candidates if c.distance == best_distance]
            chosen = sorted(best, key=lambda item: (-item.prefix_len, item.word))[0]
            status = "corrige-levenshtein"

        results.append({
            "token": token,
            "status": status,
            "correction": chosen.word,
            "lemma": chosen.lemma,
            "candidates": [c.word for c in candidates],
        })

    return results


def format_results(results: Iterable[dict[str, object]]) -> str:
    """Formate les résultats d'analyse en une chaîne lisible.

    Args:
        results (Iterable[dict[str, object]]): Résultats de `analyze_query`.

    Returns:
        str: Représentation textuelle, un token par ligne.
    """
    lines = []
    for result in results:
        lines.append(
            f"- token={result['token']} | statut={result['status']}"
            f" | correction={result['correction']} | lemme={result['lemma']}"
            f" | candidats={result['candidates']}"
        )
    return "\n".join(lines)


def main() -> int:
    """Lance la boucle interactive de correction de requêtes.

    Returns:
        int: Code de sortie (0 = succès, 1 = lexique introuvable).
    """
    lexicon_path = DEFAULT_FULL_LEXICON

    if not lexicon_path.exists():
        print(f"Lexique introuvable : {lexicon_path}", file=sys.stderr)
        return 1

    print(f"Chargement du lexique depuis : {lexicon_path}...")
    lexicon = Lexicon.from_tsv(lexicon_path)

    seuil_min = 2
    seuil_max = 3
    seuil_proximite = 0.4

    while True:
        query = input("\nSaisissez une requête (ou Entrée pour quitter) : ").strip()
        if not query:
            break

        results = analyze_query(
            query=query,
            lexicon=lexicon,
            seuil_min=seuil_min,
            seuil_max=seuil_max,
            seuil_proximite=seuil_proximite,
        )

        print(f"\n--- Résultats pour : '{query}' ---")
        print(format_results(results))
        print("-" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
