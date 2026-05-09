"""Pipeline de traitement des requêtes en langage naturel pour le moteur TD6.

Analyse une requête textuelle en français et en extrait une représentation
structurée : mots-clés (forme DNF labellisée sur titre/contenu), opérateurs
logiques, filtres de rubrique, filtres de date et contraintes d'images.

Le modèle DNF labellisé (``key_word_groups``) est une liste de groupes, chacun
étant un dict ``{"title": [...], "content": [...]}``. Au sein d'un groupe, les
contraintes sont combinées en ET ; les groupes sont combinés en OU. Les
contraintes ``title`` ciblent la zone `titre` de l'index inverse ; les
contraintes ``content`` ciblent les zones `titre` et `texte`.
"""

import csv
import pickle
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import spacy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    spacy = None

MONTH = r"janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre"
DMY_PATTERN_NUMERO = r"\b(?P<day>\d{1,2})[/\-\s]+(?P<month>\d{1,2})[/\-\s]+(?P<year>\d{4})\b"
DMY_PATTERN_TEXTE = rf"\b(?P<day>\d{{1,2}})\s+(?P<month_name>{MONTH})\s+(?P<year>\d{{4}})\b"
MY_PATTERN = r"\b(?P<month>\d{1,2})[/\-\s]+(?P<year>\d{4})\b"
MY_TEXT = rf"\b(?P<month_name>{MONTH})\s+(?P<year>\d{{4}})\b"
Y_PATTERN = r"\b(?P<year>19\d{2}|20\d{2})\b"
DMY_PATTERN_NUMERO_RAW = r"\b\d{1,2}[/\-\s]+\d{1,2}[/\-\s]+\d{4}\b"
DMY_PATTERN_TEXTE_RAW = rf"\b\d{{1,2}}\s+(?:{MONTH})\s+\d{{4}}\b"
MY_PATTERN_RAW = r"\b\d{1,2}[/\-\s]+\d{4}\b"
MY_TEXT_RAW = rf"\b(?:{MONTH})\s+\d{{4}}\b"
Y_PATTERN_RAW = r"\b(?:19\d{2}|20\d{2})\b"
DE_VARIANTS_PATTERN = r"(?:de la|de l[''']|de|du|des|d['''])"

# Capture la rubrique jusqu'à une conjonction logique ou la fin.
# Le préfixe « est » est ignoré : « rubrique est Focus » → « Focus ».
# S'arrête aux conjonctions et/ou/sans/mais/dont/qui pour ne pas capturer
# le reste de la phrase.
RUBRIQUE_PATTERN = re.compile(
    r"\b(?:(?:dans|sur|pour|provenant|de|du|des|d[''])\s+)?(?:la|le|les|l[''])?\s*\brubrique\s+"
    r"(?:est\s+|se\s+nomme\s+)?(?P<rubrique>.+?)"
    r"(?=$|\b(?:et|ou|sans|mais|dont|qui|parlant|traitant|mentionnant|contenant|évoquant"
    r"|impliquant|portant|provenant|datant)\b)",
    re.IGNORECASE,
)
BETWEEN_DMY = re.compile(
    rf"\bentre\s+(?:le\s+)?(?P<start>{DMY_PATTERN_NUMERO_RAW}|{DMY_PATTERN_TEXTE_RAW})"
    rf"\s+et\s+(?:le\s+)?(?P<end>{DMY_PATTERN_NUMERO_RAW}|{DMY_PATTERN_TEXTE_RAW})\b",
    re.IGNORECASE,
)
BETWEEN_MY = re.compile(
    rf"\bentre\s+(?P<start>{MY_PATTERN_RAW}|{MY_TEXT_RAW})\s+et\s+(?P<end>{MY_PATTERN_RAW}|{MY_TEXT_RAW})\b",
    re.IGNORECASE,
)
BETWEEN_Y = re.compile(
    rf"\bentre\s+(?P<start>{Y_PATTERN_RAW})\s+et\s+(?P<end>{Y_PATTERN_RAW})\b",
    re.IGNORECASE,
)
OP_PATTERN = re.compile(r"(?P<part1>.+?)\b(?P<op>et|ou|sans)\b(?P<part2>.+)", re.IGNORECASE)
LOGICAL_OP_PATTERN = re.compile(
    r"\bet\s+non\s+pas\b|\bnon\s+pas\b|\bmais\s+pas\b|\bsans\b|\bet\b|\bou\b",
    re.IGNORECASE,
)
TITLE_CONTAINS_PATTERN = re.compile(
    r"\bdont\s+le\s+titre\s+(?:contient|évoque)\s+(?:le\s+mot|les\s+mots|le\s+terme)?\s*"
    r"\"?(?P<value>[^\"]+?)\"?(?:$|\b(?:et|ou|mais\s+pas|sans)\b)",
    re.IGNORECASE,
)
IMAGE_PATTERN = re.compile(
    r"\bavec\s+des?\s+images?\b|\bavec\s+image\b|\bcontenant\s+une?\s+image\b"
    r"|\bqui\s+ont\s+des?\s+images?\b",
    re.IGNORECASE,
)
WITHOUT_IMAGE_PATTERN = re.compile(r"\bsans\s+image\b|\bsans\s+images\b", re.IGNORECASE)
NEGATIVE_THEME_PATTERN = re.compile(
    rf"\bn['']?e?\s*(?:parl(?:e|ent|ait|aient|ant|er)|trait(?:e|ent|ait|aient|ant|er)"
    rf"|évoqu(?:e|ent|ait|aient|ant|er)|mentionn(?:e|ent|ait|aient|ant|er)"
    rf"|concern(?:e|ent|ait|aient)|port(?:e|ent|ait|aient|ant|er))\s+pas\s+"
    rf"{DE_VARIANTS_PATTERN}(?P<theme>.+)",
    re.IGNORECASE,
)
THEME_TRIGGER_PATTERN = re.compile(
    rf"\b(?:parl(?:e|ent|ant|er)(?:\s+{DE_VARIANTS_PATTERN})?|trait(?:e|ant|er)\s+{DE_VARIANTS_PATTERN}"
    rf"|sur|a\s+propos\s+{DE_VARIANTS_PATTERN}|évoqu(?:e|ent|ant|er)|mentionn(?:e|ent|ant|er)"
    rf"|port(?:e|ent|ant|er)\s+sur|li(?:e|es)\s+a|concern(?:e|ent)|contien(?:t|nent)"
    rf"|contenant|possèd[e]?(?:nt)?|possédant|impliqu(?:e|ent|ant|er)"
    rf"|comport(?:e|ent|ant|er))\b\s*(?P<theme>.+)",
    re.IGNORECASE,
)

PATTERNS = {
    "dmy_numero": re.compile(DMY_PATTERN_NUMERO, re.IGNORECASE),
    "dmy_text": re.compile(DMY_PATTERN_TEXTE, re.IGNORECASE),
    "my_numero": re.compile(MY_PATTERN, re.IGNORECASE),
    "my_text": re.compile(MY_TEXT, re.IGNORECASE),
    "y": re.compile(Y_PATTERN, re.IGNORECASE),
    "between_dmy": BETWEEN_DMY,
    "between_my": BETWEEN_MY,
    "between_y": BETWEEN_Y,
    "rubrique": RUBRIQUE_PATTERN,
    "op": OP_PATTERN,
}

TD5_DIR = Path(__file__).resolve().parent
ROOT_DIR = TD5_DIR.parent

LEMMATISATION_FILE = ROOT_DIR / "TD3" / "mot_lemma_list.txt"
PICKELE_LEMMA_FILE = TD5_DIR / "lemma_dict.pkl"
TF_IDF_FILE = ROOT_DIR / "TD3" / "tf-idf.txt"
PICKLE_TF_IDF_FILE = TD5_DIR / "tf-idf.pkl"
ANTI_DICT_FILE = ROOT_DIR / "TD3" / "anti_dict.txt"
PICKLE_ANTI_LIST = TD5_DIR / "anti_list.pkl"
RUBRIQUE_FILE = ROOT_DIR / "TD3" / "corpus_filtre.xml"

_NLP = None
_LEMMA_DICT = None
_RUBRIQUES = None


def get_nlp():
    """Charge le modèle spaCy une seule fois (singleton).

    Returns:
        spacy.language.Language | None: Pipeline spaCy, ou ``None`` si spaCy
        est indisponible.
    """
    global _NLP
    if _NLP is not None:
        return _NLP
    if spacy is None:
        _NLP = None
        return None
    _NLP = spacy.load("fr_core_news_sm")
    return _NLP


def get_lemma_dict() -> dict[str, str]:
    """Charge le dictionnaire de lemmatisation (pickle ou TSV).

    Tente d'abord le fichier pickle mis en cache ; sinon lit le fichier TSV
    source et retourne le dictionnaire ``{forme_fléchie: lemme}``.

    Returns:
        dict[str, str]: Dictionnaire de lemmatisation.
    """
    global _LEMMA_DICT
    if _LEMMA_DICT is not None:
        return _LEMMA_DICT
    if PICKELE_LEMMA_FILE.exists():
        with open(PICKELE_LEMMA_FILE, "rb") as f:
            _LEMMA_DICT = pickle.load(f)
            return _LEMMA_DICT
    lemma_dict: dict[str, str] = {}
    if LEMMATISATION_FILE.exists():
        with open(LEMMATISATION_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for raw in reader:
                if not raw or len(raw) < 2:
                    continue
                inflection = raw[0].strip()
                lemma = raw[1].strip()
                if inflection:
                    lemma_dict[inflection.lower()] = lemma.lower()
    _LEMMA_DICT = lemma_dict
    return _LEMMA_DICT


def get_rubriques_corpus() -> list[str]:
    """Renvoie la liste des rubriques connues, triées par longueur décroissante.

    Charge les rubriques depuis le corpus filtré (XML) et met en cache le
    résultat. Le tri par longueur décroissante garantit que la correspondance
    préfixale choisit la rubrique la plus spécifique en premier.

    Returns:
        list[str]: Rubriques en minuscules, de la plus longue à la plus courte.
    """
    global _RUBRIQUES
    if _RUBRIQUES is not None:
        return _RUBRIQUES
    if not RUBRIQUE_FILE.exists():
        _RUBRIQUES = []
        return _RUBRIQUES
    _RUBRIQUES = [r.lower() for r in get_rubrique(RUBRIQUE_FILE)]
    _RUBRIQUES.sort(key=len, reverse=True)
    return _RUBRIQUES


def load_lemmatisatioin_file():
    """Génère les fichiers pickle de lemmes, TF-IDF et anti-dictionnaire.

    Lit les fichiers sources TSV/TXT et sérialise les structures de données
    en fichiers `.pkl` dans le dossier TD5 pour accélérer les chargements
    ultérieurs.
    """
    TD5_DIR.mkdir(parents=True, exist_ok=True)

    lemma_dict: dict[str, str] = {}
    with open(LEMMATISATION_FILE, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for raw in reader:
            if not raw or len(raw) < 2:
                continue
            inflection = raw[0].strip()
            lemma = raw[1].strip()
            if inflection:
                lemma_dict[inflection.lower()] = lemma.lower()
    with open(PICKELE_LEMMA_FILE, "wb") as file:
        pickle.dump(lemma_dict, file)

    content = TF_IDF_FILE.read_text(encoding="utf-8")
    lines = content.split()
    word_tf_idf: dict[str, float] = {}
    for i in range(0, len(lines) - 1, 2):
        word = lines[i].strip().lower()
        try:
            tf_idf = float(lines[i + 1])
        except ValueError:
            continue
        if word:
            word_tf_idf[word] = tf_idf
    with open(PICKLE_TF_IDF_FILE, "wb") as file:
        pickle.dump(word_tf_idf, file)

    anti_words: list[str] = []
    with open(ANTI_DICT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for raw in reader:
            if not raw:
                continue
            w = raw[0].strip().lower()
            if w:
                anti_words.append(w)
    with open(PICKLE_ANTI_LIST, "wb") as file:
        pickle.dump(anti_words, file)

    global _LEMMA_DICT
    _LEMMA_DICT = lemma_dict


def get_rubrique(file_xml: Path) -> list[str]:
    """Extrait la liste des rubriques distinctes d'un corpus XML.

    Args:
        file_xml (Path): Chemin vers le fichier XML du corpus.

    Returns:
        list[str]: Liste des rubriques uniques (ordre non garanti).
    """
    tree = ET.parse(file_xml)
    root = tree.getroot()
    rubriques = set()
    for doc in root.findall("document"):
        elem = doc.find("rubrique")
        if elem is not None and elem.text:
            rubriques.add(elem.text.strip())
    return list(rubriques)


def normaliser_texte(source: str, key_word_traite=False):
    """Normalise le texte d'une requête et extrait optionnellement les majuscules.

    Supprime les `?`, compacte les espaces. Si ``key_word_traite`` est ``True``,
    retourne également la liste des mots entièrement en majuscules (acronymes
    comme CNRS) présents dans la requête.

    Args:
        source (str): Texte brut de la requête.
        key_word_traite (bool): Si ``True``, extrait aussi les mots en majuscules.

    Returns:
        str | tuple[str, list[str]]: Texte normalisé, ou ``(texte, mots_majuscules)``
        si ``key_word_traite`` est ``True``.
    """
    source = source.replace("?", " ")
    source = re.sub(r"\s+", " ", source)

    words = re.findall(r"\b\w+\b", source)
    key_word = []

    if key_word_traite:
        for word in words:
            if any(c.isalpha() for c in word) and word.isupper():
                key_word.append(word)
        return source.strip(), key_word
    return source.strip()


def pipeline_traitement_requete(
    source: str,
    anti_list: list,
    upper_key_word: list,
) -> dict:
    """Exécute le pipeline complet de traitement d'une requête normalisée.

    Applique dans l'ordre : extraction des filtres globaux, suppression du
    préfixe de type de requête, découpage sur les opérateurs logiques,
    extraction des filtres structurels (DNF labellisée sur titre/contenu),
    lemmatisation des mots-clés.

    Le résultat est un dictionnaire épars dont les clés non vides sont parmi :
    ``key_word``, ``key_word_exclu``, ``key_word_groups``, ``title_keywords``,
    ``rubrique``, ``date``, ``image``.

    Args:
        source (str): Requête déjà normalisée (sans `?`, espaces compactés).
        tf_idf_dict (dict): Scores TF-IDF (non utilisés directement ici, transmis
            au pipeline pour extension future).
        anti_list (list): Liste des tokens à exclure (anti-dictionnaire).
        upper_key_word (list[str]): Mots entièrement en majuscules extraits avant
            normalisation (acronymes comme CNRS).

    Returns:
        dict: Représentation structurée de la requête.
    """
    # Étape 1 : extraction des filtres globaux (rubrique, date, image)
    source, filtres = extraire_filtres_globaux(source)

    # Étape 2 : suppression du préfixe de type requête (« les articles… »)
    source = supprimer_prefixe_avant_articles(source)

    # Étape 3 : découpage sur les opérateurs logiques
    parts, operateurs = decouper_expression_logique(normaliser_texte(source))
    if not parts:
        parts = [normaliser_texte(source)]

    # Étape 4 : extraction des filtres structurels (DNF labellisée titre/contenu)
    parts_restants, themes_groups, themes_exclu, image = traiter_filtres_structurels(
        parts, operateurs
    )
    if image is not None:
        filtres["image"] = image

    # Étape 5 : lemmatisation des mots-clés et construction de la DNF finale
    key_word, key_word_exclu, title_keywords, key_word_groups = traiter_mots_cles(
        source, parts_restants, themes_groups, themes_exclu, anti_list, upper_key_word
    )

    # Étape 6 : assemblage du résultat (uniquement les champs non vides)
    result: dict = {"key_word": key_word, "key_word_exclu": key_word_exclu}
    if key_word_groups:
        result["key_word_groups"] = key_word_groups
    if title_keywords:
        result["title_keywords"] = title_keywords
    if filtres["rubrique"] is not None:
        result["rubrique"] = filtres["rubrique"]
    if filtres["date"] is not None:
        result["date"] = filtres["date"]
    if filtres["image"] is not None:
        result["image"] = filtres["image"]
    return result


def supprimer_prefixe_avant_articles(source: str) -> str:
    """Supprime le préfixe de type requête jusqu'au mot « article(s) ».

    Args:
        source (str): Texte de la requête.

    Returns:
        str: Texte après « article(s) », ou texte d'origine si absent.
    """
    match = re.search(r"\barticles?\b", source, re.IGNORECASE)
    if match is None:
        return source.strip()
    return source[match.end():].strip(" ,")


def masquer_intervalles_temporels(source: str) -> str:
    """Remplace les intervalles de dates par des espaces pour isoler les opérateurs logiques.

    Évite que « entre X et Y » soit interprété comme un opérateur logique « et ».

    Args:
        source (str): Texte de la requête.

    Returns:
        str: Texte masqué (même longueur que l'original).
    """
    source_masque = source
    for pattern_name in ("between_dmy", "between_my", "between_y"):
        pattern = PATTERNS[pattern_name]
        source_masque = pattern.sub(lambda m: " " * (m.end() - m.start()), source_masque)
    return source_masque


def extraire_filtres_globaux(source: str) -> tuple[str, dict]:
    """Extrait les filtres globaux (rubrique, date, image) et les retire du texte.

    L'extraction respecte une priorité décroissante pour les dates :
    ``between_dmy`` > ``between_my`` > ``between_y`` > ``dmy`` > ``my`` > ``y``.

    Args:
        source (str): Texte brut de la requête.

    Returns:
        tuple[str, dict]: Texte nettoyé et dictionnaire
        ``{"rubrique": ..., "date": ..., "image": ...}``.
    """
    filtres: dict = {"rubrique": None, "date": None, "image": None}

    # Image
    m = WITHOUT_IMAGE_PATTERN.search(source)
    if m:
        filtres["image"] = False
        source = source[: m.start()] + source[m.end():]
    else:
        m = IMAGE_PATTERN.search(source)
        if m:
            filtres["image"] = True
            source = source[: m.start()] + source[m.end():]

    # Dates (priorité décroissante)
    _dmy_num_re = re.compile(DMY_PATTERN_NUMERO_RAW, re.IGNORECASE)
    _dmy_txt_re = re.compile(DMY_PATTERN_TEXTE_RAW, re.IGNORECASE)
    _my_num_re = re.compile(MY_PATTERN_RAW, re.IGNORECASE)
    _my_txt_re = re.compile(MY_TEXT_RAW, re.IGNORECASE)
    _y_re = re.compile(Y_PATTERN_RAW, re.IGNORECASE)

    date_patterns_ordered = [
        (BETWEEN_DMY, lambda m: {"type": "between_dmy", "start": m.group("start"), "end": m.group("end")}),
        (BETWEEN_MY,  lambda m: {"type": "between_my",  "start": m.group("start"), "end": m.group("end")}),
        (BETWEEN_Y,   lambda m: {"type": "between_y",   "start": m.group("start"), "end": m.group("end")}),
        (_dmy_num_re, lambda m: {"type": "dmy", "value": m.group(0)}),
        (_dmy_txt_re, lambda m: {"type": "dmy", "value": m.group(0)}),
        (_my_num_re,  lambda m: {"type": "my",  "value": m.group(0)}),
        (_my_txt_re,  lambda m: {"type": "my",  "value": m.group(0)}),
        (_y_re,       lambda m: {"type": "y",   "value": m.group(0)}),
    ]

    for pattern, extractor in date_patterns_ordered:
        m = pattern.search(source)
        if m:
            filtres["date"] = extractor(m)
            break

    # Rubrique : extraire la valeur de la première occurrence, puis supprimer toutes
    m = RUBRIQUE_PATTERN.search(source)
    if m:
        rubrique_raw = nettoyer_valeur_extraite(m.group("rubrique")).lower()
        if rubrique_raw:
            rubriques_connues = get_rubriques_corpus()
            rubrique_finale = None
            rubrique_raw_norm = _strip_accents(rubrique_raw)
            for r in rubriques_connues:
                if rubrique_raw_norm.startswith(_strip_accents(r)):
                    rubrique_finale = r
                    break
            filtres["rubrique"] = rubrique_finale or rubrique_raw
        source = RUBRIQUE_PATTERN.sub("", source)

    source = re.sub(r"\s+", " ", source).strip()
    return source, filtres


def decouper_expression_logique(source: str) -> tuple[list[str], list[str]]:
    """Décompose récursivement une expression sur les opérateurs logiques.

    Les intervalles de dates sont masqués avant la recherche d'opérateurs pour
    éviter les faux positifs sur « entre X et Y ».

    Args:
        source (str): Texte normalisé de la requête (ou d'une sous-expression).

    Returns:
        tuple[list[str], list[str]]: Paire ``(parties, opérateurs)``.
    """
    source = source.strip()
    if not source:
        return [], []

    source_masque = masquer_intervalles_temporels(source)
    op_match = LOGICAL_OP_PATTERN.search(source_masque)
    if op_match is None:
        return [source.strip()], []

    part1 = source[:op_match.start()].strip(" ,")
    part2 = source[op_match.end():].strip(" ,")
    operateur = op_match.group(0).lower()

    parties_gauche, operateurs_gauche = decouper_expression_logique(part1)
    parties_droite, operateurs_droite = decouper_expression_logique(part2)

    return (
        parties_gauche + parties_droite,
        operateurs_gauche + [operateur] + operateurs_droite,
    )


def nettoyer_valeur_extraite(value: str) -> str:
    """Supprime les espaces, virgules et guillemets en début et fin de valeur.

    Args:
        value (str): Valeur brute extraite par une expression régulière.

    Returns:
        str: Valeur nettoyée.
    """
    value = value.strip(" ,\"'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _strip_accents(s: str) -> str:
    """Supprime les accents d'une chaîne pour la comparaison insensible aux accents."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if not unicodedata.combining(c))


def supprimer_prefixe_theme(theme: str) -> str:
    """Supprime les articles et préfixes de thème en tête d'une chaîne.

    Args:
        theme (str): Thème brut extrait par un patron regex.

    Returns:
        str: Thème débarrassé de ses préfixes.
    """
    theme = nettoyer_valeur_extraite(theme)
    # Supprimer « les mots X / le mot X » avant le dépouillement d'article
    theme = re.sub(r"^(?:les?\s+mots?\b|les?\s+termes?\b)\s*", "", theme, flags=re.IGNORECASE)
    theme = re.sub(
        r"^(des|du|de la|de l['']|d['']|de|les|la|le|l[''])\s*",
        "",
        theme,
        flags=re.IGNORECASE,
    )
    # Supprimer « le/les mot(s)/terme(s) » en milieu de phrase
    theme = re.sub(r"\bles?\s+mots?\s+|\bles?\s+termes?\s+", " ", theme, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", theme).strip()


def est_partie_temporelle(partie: str) -> bool:
    """Détermine si une partie de la requête est une contrainte temporelle.

    Args:
        partie (str): Fragment de la requête à tester.

    Returns:
        bool: ``True`` si la partie contient ou décrit une date ou une plage.
    """
    partie = nettoyer_valeur_extraite(partie.lower())
    if not partie:
        return False

    if (
        PATTERNS["between_dmy"].search(partie)
        or PATTERNS["between_my"].search(partie)
        or PATTERNS["between_y"].search(partie)
    ):
        return True

    if PATTERNS["dmy_numero"].search(partie) or PATTERNS["dmy_text"].search(partie):
        return True

    if PATTERNS["my_numero"].search(partie) or PATTERNS["my_text"].search(partie):
        return True

    if PATTERNS["y"].search(partie):
        mots_temporels = ("mois", "annee", "an", "apres", "avant", "depuis", "partir", "date", "publie")
        if any(re.search(r'\b' + mot + r'\b', partie) for mot in mots_temporels):
            return True

    return False


def extraire_theme_depuis_part_v2(partie: str, dernier_type: str) -> str | None:
    """Extrait le thème d'une partie de requête, si elle en contient un.

    Args:
        partie (str): Fragment de requête à analyser.
        dernier_type (str | None): Type de la partie précédente
            (``"theme"``, ``"theme_exclu"``, ``"title"`` ou ``None``).

    Returns:
        str | None: Thème extrait et nettoyé, ou ``None`` si absent.
    """
    partie = partie.strip()
    if not partie:
        return None
    if est_partie_temporelle(partie):
        return None

    partie = re.sub(r"^(?:non\s+pas|pas)\s+", "", partie, flags=re.IGNORECASE)

    match = THEME_TRIGGER_PATTERN.search(partie)
    if match is not None:
        theme = supprimer_prefixe_theme(match.group("theme"))
        return theme if theme else None

    if dernier_type in {"theme", "theme_exclu"}:
        match_simple = re.search(
            r"\b(?:des|de l[''']|de la|de|du|d['''])\s*(.+)$", partie, re.IGNORECASE
        )
        if match_simple is not None:
            theme = supprimer_prefixe_theme(match_simple.group(1))
            return theme if theme else None

    return nettoyer_valeur_extraite(partie)


def extraire_theme_negatif_depuis_part(partie: str) -> str | None:
    """Extrait le thème d'une contrainte négative (« ne parle pas de… »).

    Args:
        partie (str): Fragment de requête potentiellement négatif.

    Returns:
        str | None: Thème négatif extrait, ou ``None`` si absent.
    """
    partie = partie.strip()
    if not partie:
        return None

    match = NEGATIVE_THEME_PATTERN.search(partie)
    if match is None:
        return None

    theme = supprimer_prefixe_theme(match.group("theme"))
    return theme if theme else None


def nettoyer_titre_suite(partie: str) -> str:
    """Nettoie une partie de titre en supprimant les connecteurs initiaux.

    Args:
        partie (str): Fragment de titre brut.

    Returns:
        str: Titre nettoyé.
    """
    partie = nettoyer_valeur_extraite(partie)
    partie = re.sub(r"^(?:et|ou)\s+", "", partie, flags=re.IGNORECASE)
    partie = re.sub(r"^(?:le|les)\s+(?:mot|mots|terme|termes)\s*", "", partie, flags=re.IGNORECASE)
    return nettoyer_valeur_extraite(partie)


def nettoyer_theme_suite(partie: str) -> str:
    """Nettoie une partie de thème en supprimant les connecteurs et préfixes.

    Args:
        partie (str): Fragment de thème brut.

    Returns:
        str: Thème nettoyé.
    """
    partie = nettoyer_valeur_extraite(partie)
    partie = re.sub(r"^(?:et|ou)\s+", "", partie, flags=re.IGNORECASE)
    partie = supprimer_prefixe_theme(partie)
    return nettoyer_valeur_extraite(partie)


def traiter_filtres_structurels(
    parts: list[str],
    operateurs: list[str],
) -> tuple[list[str], list[dict], list[str], bool | None]:
    """Classe chaque partie de requête en groupe DNF, exclusion ou résidu.

    Construit une DNF labellisée (``themes_groups``) sur les contraintes de
    titre et de contenu. Un nouveau groupe est ouvert quand l'opérateur
    précédant une clause positive est ``ou``. Les clauses négatives alimentent
    ``themes_exclus``. Les parties temporelles et non reconnues sont renvoyées
    dans ``parts_restantes``.

    Args:
        parts (list[str]): Parties de la requête découpées sur les opérateurs.
        operateurs (list[str]): Opérateurs entre les parties
            (``"et"``, ``"ou"``, ``"sans"``, etc.).

    Returns:
        tuple: ``(parts_restantes, themes_groups, themes_exclus, image)`` où
        ``themes_groups`` est une liste de dicts ``{"title": [...], "content": [...]}``.
    """
    parts_restantes: list[str] = []
    themes_groups: list[dict] = []
    current_group: dict = {"title": [], "content": []}
    themes_exclus: list[str] = []
    image: bool | None = None
    dernier_type = None

    def group_non_vide() -> bool:
        return bool(current_group["title"] or current_group["content"])

    def flush_group() -> None:
        nonlocal current_group
        if group_non_vide():
            themes_groups.append(current_group)
            current_group = {"title": [], "content": []}

    def maybe_flush_for_or(est_ou: bool) -> None:
        if est_ou and group_non_vide():
            flush_group()

    for i, part in enumerate(parts):
        part_courante = part.strip()
        if not part_courante:
            continue

        operateur_precedent = operateurs[i - 1] if i > 0 and i - 1 < len(operateurs) else None
        est_negatif = operateur_precedent in {"sans", "mais pas", "non pas", "et non pas"}
        est_ou = operateur_precedent == "ou"

        if est_partie_temporelle(part_courante):
            parts_restantes.append(part_courante)
            dernier_type = None
            continue

        title_match = TITLE_CONTAINS_PATTERN.search(part_courante)
        if title_match is not None:
            titre = nettoyer_valeur_extraite(title_match.group("value"))
            if titre:
                maybe_flush_for_or(est_ou)
                current_group["title"].append(titre)
            dernier_type = "title"
            continue

        theme_negatif = extraire_theme_negatif_depuis_part(part_courante)
        if theme_negatif:
            themes_exclus.append(theme_negatif)
            dernier_type = "theme_exclu"
            continue

        theme = extraire_theme_depuis_part_v2(part_courante, dernier_type)
        if theme and theme != part_courante:
            if est_negatif:
                themes_exclus.append(theme)
                dernier_type = "theme_exclu"
            else:
                maybe_flush_for_or(est_ou)
                current_group["content"].append(theme)
                dernier_type = "theme"
            continue

        if dernier_type == "title" and operateur_precedent in {"et", "ou"}:
            titre_suite = nettoyer_titre_suite(part_courante)
            if titre_suite:
                maybe_flush_for_or(est_ou)
                current_group["title"].append(titre_suite)
                continue
            dernier_type = None

        if dernier_type == "theme" and operateur_precedent in {"et", "ou"}:
            theme_suite = nettoyer_theme_suite(part_courante)
            if theme_suite:
                maybe_flush_for_or(est_ou)
                current_group["content"].append(theme_suite)
                continue
            dernier_type = None

        if dernier_type == "theme_exclu" and operateur_precedent in {"et", "ou"}:
            theme_suite = nettoyer_theme_suite(part_courante)
            if theme_suite:
                themes_exclus.append(theme_suite)
                continue
            dernier_type = None

        if theme and est_negatif:
            themes_exclus.append(theme)
            dernier_type = "theme_exclu"
        else:
            parts_restantes.append(part_courante)
            dernier_type = None

    flush_group()
    return parts_restantes, themes_groups, themes_exclus, image


def extraire_keywords_partie(partie: str, anti_list: list) -> list[str]:
    """Lemmatise et filtre les mots-clés d'un fragment de requête.

    Utilise spaCy si disponible, sinon le dictionnaire de lemmatisation mis en
    cache. Élimine les tokens d'une seule lettre et les mots de l'anti-dict.

    Args:
        partie (str): Fragment de texte à analyser.
        anti_list (list): Tokens à exclure.

    Returns:
        list[str]: Lemmes uniques conservés, dans l'ordre d'apparition.
    """
    nlp = get_nlp()
    if nlp is not None:
        doc = nlp(partie)
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
    else:
        lemma_dict = get_lemma_dict()
        tokens = [lemma_dict.get(w, w) for w in re.findall(r"[A-Za-zÀ-ÿ]+", partie.lower())]

    seen: set[str] = set()
    result: list[str] = []
    for w in tokens:
        if len(w) <= 1 or w in anti_list or w in seen:
            continue
        seen.add(w)
        result.append(w)
    return result


def traiter_mots_cles(
    source: str,
    parts_restants: list[str],
    themes_groups: list[dict],
    themes_exclus: list[str],
    anti_list: list,
    upper_key_word: list[str],
) -> tuple[list[str], list[str], list[str], list[dict]]:
    """Lemmatise les mots-clés et assemble la représentation DNF finale.

    Convertit les groupes bruts (chaînes thème/titre) en groupes lemmatisés.
    Les clauses ``title`` sont conservées en minuscules sans lemmatisation
    (elles ciblent la zone `titre` de l'index inverse, qui stocke les formes
    brutes). Les clauses ``content`` sont lemmatisées via
    ``extraire_keywords_partie``.

    Les acronymes en majuscules (``upper_key_word``) sont distribués en
    contrainte ET globale dans chaque groupe ; si aucun groupe n'existe, un
    groupe singleton est créé.

    ``key_word`` est l'union plate ordonnée de tous les lemmes de contenu ;
    utilisé par le code de génération d'extraits. ``title_keywords`` est
    l'union plate des formes de titre ; maintenu pour la compatibilité
    ascendante avec ``app.py`` et ``evaluation.py``.

    Args:
        source (str): Texte normalisé de la requête (non utilisé directement,
            présent pour une extension future).
        parts_restants (list[str]): Parties non classifiées par
            ``traiter_filtres_structurels``.
        themes_groups (list[dict]): Groupes DNF bruts
            ``{"title": [...], "content": [...]}``.
        themes_exclus (list[str]): Thèmes à exclure.
        anti_list (list): Anti-dictionnaire.
        upper_key_word (list[str]): Mots en majuscules (acronymes).

    Returns:
        tuple[list[str], list[str], list[str], list[dict]]: Quadruplet
        ``(key_word, key_word_exclu, title_keywords, key_word_groups)``.
    """
    upper_normalises: list[str] = []
    seen_upper: set[str] = set()
    for word in upper_key_word:
        wn = word.lower()
        if len(wn) <= 1 or wn in seen_upper:
            continue
        seen_upper.add(wn)
        upper_normalises.append(wn)

    key_word_groups: list[dict] = []
    for grp in themes_groups:
        title_kws: list[str] = []
        seen_t: set[str] = set()
        for raw in grp.get("title", []):
            t_norm = raw.lower().strip()
            if not t_norm or t_norm in seen_t:
                continue
            seen_t.add(t_norm)
            title_kws.append(t_norm)

        content_kws: list[str] = []
        seen_c: set[str] = set()
        for theme in grp.get("content", []):
            for w in extraire_keywords_partie(theme, anti_list):
                if w in seen_c:
                    continue
                seen_c.add(w)
                content_kws.append(w)

        if title_kws or content_kws:
            key_word_groups.append({"title": title_kws, "content": content_kws})

    if upper_normalises:
        if not key_word_groups:
            key_word_groups = [{"title": [], "content": list(upper_normalises)}]
        else:
            for grp in key_word_groups:
                grp_set = set(grp["content"])
                for w in upper_normalises:
                    if w in grp_set:
                        continue
                    grp["content"].append(w)
                    grp_set.add(w)

    key_word: list[str] = []
    deja_kw: set[str] = set()
    title_keywords: list[str] = []
    deja_t: set[str] = set()
    for grp in key_word_groups:
        for w in grp["content"]:
            if w not in deja_kw:
                key_word.append(w)
                deja_kw.add(w)
        for w in grp["title"]:
            if w not in deja_t:
                title_keywords.append(w)
                deja_t.add(w)

    key_word_exclu: list[str] = []
    exclu_vus: set[str] = set()
    for theme in themes_exclus:
        for w in extraire_keywords_partie(theme, anti_list):
            if w not in exclu_vus:
                exclu_vus.add(w)
                key_word_exclu.append(w)

    return key_word, key_word_exclu, title_keywords, key_word_groups


if __name__ == "__main__":
    file_name = Path(__file__).parent / "requete.txt"
    load_lemmatisatioin_file()

    with open(file_name, "r", encoding="utf-8") as f:
        requests = f.readlines()
    with open(PICKLE_TF_IDF_FILE, "rb") as file:
        tf_idf_dict = pickle.load(file)
    with open(PICKLE_ANTI_LIST, "rb") as file:
        anti_list = pickle.load(file)

    for request in requests:
        request_norm, upper_kw = normaliser_texte(request, key_word_traite=True)
        result = pipeline_traitement_requete(request_norm, anti_list, upper_kw)
        if any(result.values()):
            print(f"\n{request.strip()}")
            for k, v in result.items():
                if v:
                    print(f"{k}: {v}")
