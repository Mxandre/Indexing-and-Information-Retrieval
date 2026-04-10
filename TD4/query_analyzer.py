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
    '''Supprime les accents d'une chaîne de caractères.'''
    normalized = unicodedata.normalize("NFKD", text)  ## transform the accent to unicode data
    return "".join(char for char in normalized if not unicodedata.combining(char)) ## combining to recognize if it is a adding unicode data


def normalize(text: str) -> str:
    '''Normalise un token en supprimant les accents, en le mettant en minuscules et en remplaçant les apostrophes.'''
    text = text.strip().lower().replace("’", "'")
    return strip_accents(text)


def tokenize(text: str) -> list[str]:
    '''Tokenise une chaîne de caractères en utilisant une expression régulière.'''
    return TOKEN_PATTERN.findall(text.replace("’", "'"))


def common_prefix_length(left: str, right: str) -> int:
    '''Calcule la longueur du préfixe commun entre deux chaînes de caractères.'''
    limit = min(len(left), len(right))
    size = 0
    while size < limit and left[size] == right[size]:
        size += 1
    return size


def levenshtein_distance(left: str, right: str) -> int:
    '''Calcule la distance de Levenshtein entre deux chaînes de caractères.'''
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
    '''Vérifie si un token est une entité spécifique : une date ou un nombre.'''
    return bool(DATE_PATTERN.match(token) or NUMBER_PATTERN.match(token))


def safe_spacy_tokenize_and_lemmatize(text: str) -> list[tuple[str, str]]:
    '''Tokenise et lemmatise une chaîne de caractères en utilisant Spacy.'''
    try:
        import spacy  # type: ignore

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
    word: str
    lemma: str
    prefix_len: int
    prefix_ratio: float
    distance: int


class Lexicon:
    def __init__(self, entries: dict[str, str], source: Path) -> None:
        self.entries = entries
        self.source = source
        self.words = sorted(entries)

    @classmethod
    def from_tsv(cls, path: Path) -> "Lexicon":
        '''Charge un lexique à partir d'un fichier TSV contenant des paires mot<tab>lemme.'''
        lemma_votes: dict[str, Counter[str]] = defaultdict(Counter)
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or "\t" not in line:
                    continue
                word, lemma = line.split("\t", maxsplit=1)
                lemma_votes[normalize(word)][normalize(lemma)] += 1  ## normalize to mas-> min

        entries = {
            word: votes.most_common(1)[0][0] # return the most common lemma in votes
            for word, votes in lemma_votes.items()
            if word
        }
        return cls(entries=entries, source=path)

    def contains(self, token: str) -> bool:
        return normalize(token) in self.entries

    def lemma_for(self, token: str) -> str | None:
        '''Retourne le lemme d'un token s'il est présent dans le lexique, sinon None.'''
        return self.entries.get(normalize(token))

    def generate_candidates(
        self,
        token: str,
        seuil_min: int,
        seuil_max: int,
        seuil_proximite: float,
    ) -> list[Candidate]:
        '''Génère une liste de candidats de correction 
        pour un token donné en fonction de prefixe commun, distance de Levenshtein et seuils.'''

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
    '''Analyse une requete en validant et corrigeant 
    les tokens en utilisant un lexique et des seuils de correction.'''

    # tokenisation et lemmatisation
    spacy_tokens = safe_spacy_tokenize_and_lemmatize(query)
    if spacy_tokens:
        tokens = [token for token, _lemma in spacy_tokens]
    else:
        tokens = tokenize(query)

    results: list[dict[str, object]] = []
    for token in tokens:
        normalized = normalize(token)
        if not normalized:
            continue

        if is_specific_entity(token):
            results.append(
                {
                    "token": token,
                    "status": "entite",
                    "correction": token,
                    "lemma": token,
                    "candidates": [],
                }
            )
            continue

        # Vérification directe dans le lexique
        direct_lemma = lexicon.lemma_for(token)
        if direct_lemma is not None:
            results.append(
                {
                    "token": token,
                    "status": "valide",
                    "correction": normalized,
                    "lemma": direct_lemma,
                    "candidates": [],
                }
            )
            continue

        # Si un mot n'est pas dans le lexique et n'est pas une entité spécifique, 
        # #générer des candidats de correction
        candidates = lexicon.generate_candidates(
            token=token,
            seuil_min=seuil_min,
            seuil_max=seuil_max,
            seuil_proximite=seuil_proximite,
        )

        if not candidates:
            results.append(
                {
                    "token": token,
                    "status": "introuvable",
                    "correction": None,
                    "lemma": None,
                    "candidates": [],
                }
            )
            continue

        if len(candidates) == 1:
            chosen = candidates[0]
            status = "corrige-candidat-unique"
        else:
            best_distance = candidates[0].distance
            best = [candidate for candidate in candidates if candidate.distance == best_distance]
            chosen = sorted(best, key=lambda item: (-item.prefix_len, item.word))[0]
            status = "corrige-levenshtein"

        results.append(
            {
                "token": token,
                "status": status,
                "correction": chosen.word,
                "lemma": chosen.lemma,
                "candidates": [candidate.word for candidate in candidates],
            }
        )

    return results


def format_results(results: Iterable[dict[str, object]]) -> str:
    '''Formate les résultats en une chaîne de caractères.'''
    lines = []
    for result in results:
        token = str(result["token"])
        status = str(result["status"])
        correction = result["correction"]
        lemma = result["lemma"]
        candidates = result["candidates"]
        lines.append(
            f"- token={token} | statut={status} | correction={correction} | lemme={lemma} | candidats={candidates}"
        )
    return "\n".join(lines)


def main() -> int:
    # Utiliser le lexique complet par défaut pour les tests
    lexicon_path = DEFAULT_FULL_LEXICON

    if not lexicon_path.exists():
        print(f"Lexique introuvable: {lexicon_path}", file=sys.stderr)
        return 1

    print(f"Chargement du lexique depuis : {lexicon_path}...")
    lexicon = Lexicon.from_tsv(lexicon_path)

    # Paramètres par défaut codés en dur
    seuil_min = 2
    seuil_max = 3
    seuil_proximite = 0.4

    while True:
        query = input("\nSaisissez une requete (ou Entrée pour quitter) : ").strip()
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
