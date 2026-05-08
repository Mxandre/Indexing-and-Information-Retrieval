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
DE_VARIANTS_PATTERN = r"(?:de la|de l['’']|de|du|des|d['’'])"

# Capture la rubrique jusqu'à une conjonction logique ou la fin.
# "est " prefix is skipped so "rubrique est Focus" → "Focus".
# Stops at "et/ou/sans/mais/dont/qui" to avoid capturing surrounding sentence.
RUBRIQUE_PATTERN = re.compile(
    r"\brubrique\s+(?:est\s+|se\s+nomme\s+)?(?P<rubrique>.+?)(?=$|\b(?:et|ou|sans|mais|dont|qui)\b)",
    re.IGNORECASE,
)
BETWEEN_DMY = re.compile(
    rf"\bentre\s+(?:le\s+)?(?P<start>{DMY_PATTERN_NUMERO_RAW}|{DMY_PATTERN_TEXTE_RAW})\s+et\s+(?:le\s+)?(?P<end>{DMY_PATTERN_NUMERO_RAW}|{DMY_PATTERN_TEXTE_RAW})\b",
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
LOGICAL_OP_PATTERN = re.compile(r"\bet\s+non\s+pas\b|\bnon\s+pas\b|\bmais\s+pas\b|\bsans\b|\bet\b|\bou\b", re.IGNORECASE)
TITLE_CONTAINS_PATTERN = re.compile(
    r"\bdont\s+le\s+titre\s+(?:contient|évoque)\s+(?:le\s+mot|les\s+mots|le\s+terme)?\s*\"?(?P<value>[^\"]+?)\"?(?:$|\b(?:et|ou|mais\s+pas|sans)\b)",
    re.IGNORECASE,
)
IMAGE_PATTERN = re.compile(r"\bavec\s+des?\s+images?\b|\bavec\s+image\b|\bcontenant\s+une?\s+image\b|\bqui\s+ont\s+des?\s+images?\b", re.IGNORECASE)
WITHOUT_IMAGE_PATTERN = re.compile(r"\bsans\s+image\b|\bsans\s+images\b", re.IGNORECASE)
NEGATIVE_THEME_PATTERN = re.compile(
    rf"\bn[’']?e?\s*(?:parl(?:e|ent|ait|aient|ant|er)|trait(?:e|ent|ait|aient|ant|er)|evoqu(?:e|ent|ait|aient|ant|er)|mentionn(?:e|ent|ait|aient|ant|er)|concern(?:e|ent|ait|aient)|port(?:e|ent|ait|aient|ant|er))\s+pas\s+{DE_VARIANTS_PATTERN}(?P<theme>.+)",
    re.IGNORECASE,
)
THEME_TRIGGER_PATTERN = re.compile(
    rf"\b(?:parl(?:e|ent|ant|er)(?:\s+{DE_VARIANTS_PATTERN})?|trait(?:e|ant|er)\s+{DE_VARIANTS_PATTERN}|sur|a\s+propos\s+{DE_VARIANTS_PATTERN}|evoqu(?:e|ent|ant|er)|mentionn(?:e|ent|ant|er)|port(?:e|ent|ant|er)\s+sur|li(?:e|es)\s+a|concern(?:e|ent)|contien(?:t|nent)|contenant|possèd[e]?(?:nt)?|possédant|impliqu(?:e|ent|ant|er)|comport(?:e|ent|ant|er))\b\s*(?P<theme>.+)",
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
    global _NLP
    if _NLP is not None:
        return _NLP
    if spacy is None:
        _NLP = None
        return None
    _NLP = spacy.load("fr_core_news_sm")
    return _NLP


def get_lemma_dict() -> dict[str, str]:
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
    tree = ET.parse(file_xml)
    root = tree.getroot()
    rubriques = set()
    for doc in root.findall("document"):
        elem = doc.find("rubrique")
        if elem is not None and elem.text:
            rubriques.add(elem.text.strip())
    return list(rubriques)


def normaliser_texte(source: str, key_word_traite=False):
    source = source.replace("?", " ")
    source = re.sub(r"\s+", " ", source)

    words = re.findall(r"\b\w+\b", source)
    key_word = []

    if key_word_traite:
        for word in words:
            if any(c.isalpha() for c in word) and word.isupper():
                key_word.append(word)
    source = source.lower()
    if key_word_traite:
        return source.strip(), key_word
    return source.strip()


def pipeline_traitement_requete(
    source: str,
    tf_idf_dict: dict,
    anti_list: list,
    upper_key_word: list,
) -> dict:
    """
    Process a query and return a clean dict with at most 6 public fields:
      key_word, key_word_exclu, title_keywords, rubrique, date, image.
    """
    # Step 1: strip global filters from source
    source, filtres = extraire_filtres_globaux(source)

    # Step 2: remove request-type prefix ("les articles…")
    source = supprimer_prefixe_avant_articles(source)

    # Step 3: split on logical operators
    parts, operateurs = decouper_expression_logique(normaliser_texte(source))
    if not parts:
        parts = [normaliser_texte(source)]

    # Step 4: extract structural filters
    parts_restants, title_kws, themes, themes_exclu, image = traiter_filtres_structurels(parts, operateurs)
    if image is not None:
        filtres["image"] = image

    # Step 5: extract keywords
    key_word, key_word_exclu = traiter_mots_cles(
        source, parts_restants, themes, themes_exclu, anti_list, upper_key_word
    )

    # Step 6: build sparse output (only non-None / non-empty optional fields)
    result: dict = {"key_word": key_word, "key_word_exclu": key_word_exclu}
    if title_kws:
        result["title_keywords"] = title_kws
    if filtres["rubrique"] is not None:
        result["rubrique"] = filtres["rubrique"]
    if filtres["date"] is not None:
        result["date"] = filtres["date"]
    if filtres["image"] is not None:
        result["image"] = filtres["image"]
    return result


def supprimer_prefixe_avant_articles(source: str) -> str:
    match = re.search(r"\barticles?\b", source, re.IGNORECASE)
    if match is None:
        return source.strip()
    return source[match.end():].strip(" ,")


def masquer_intervalles_temporels(source: str) -> str:
    source_masque = source
    for pattern_name in ["between_dmy", "between_my", "between_y"]:
        pattern = PATTERNS[pattern_name]
        source_masque = pattern.sub(lambda m: " " * (m.end() - m.start()), source_masque)
    return source_masque


def extraire_filtres_globaux(source: str) -> tuple[str, dict]:
    """
    Extrait les filtres globaux (rubrique, date, image) du texte source et les supprime.
    Retourne (source_nettoyé, {rubrique, date, image}).
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

    # Dates (priorité : between_dmy > between_my > between_y > dmy > my > y)
    _dmy_num_re = re.compile(DMY_PATTERN_NUMERO_RAW, re.IGNORECASE)
    _dmy_txt_re = re.compile(DMY_PATTERN_TEXTE_RAW, re.IGNORECASE)
    _my_num_re  = re.compile(MY_PATTERN_RAW, re.IGNORECASE)
    _my_txt_re  = re.compile(MY_TEXT_RAW, re.IGNORECASE)
    _y_re       = re.compile(Y_PATTERN_RAW, re.IGNORECASE)

    date_patterns_ordered = [
        (BETWEEN_DMY,  lambda m: {"type": "between_dmy", "start": m.group("start"), "end": m.group("end")}),
        (BETWEEN_MY,   lambda m: {"type": "between_my",  "start": m.group("start"), "end": m.group("end")}),
        (BETWEEN_Y,    lambda m: {"type": "between_y",   "start": m.group("start"), "end": m.group("end")}),
        (_dmy_num_re,  lambda m: {"type": "dmy", "value": m.group(0)}),
        (_dmy_txt_re,  lambda m: {"type": "dmy", "value": m.group(0)}),
        (_my_num_re,   lambda m: {"type": "my",  "value": m.group(0)}),
        (_my_txt_re,   lambda m: {"type": "my",  "value": m.group(0)}),
        (_y_re,        lambda m: {"type": "y",   "value": m.group(0)}),
    ]

    for pattern, extractor in date_patterns_ordered:
        m = pattern.search(source)
        if m:
            filtres["date"] = extractor(m)
            ## to make theme extraction cleaner, 
            # source = source[: m.start()] + source[m.end():]
            break

    # Rubrique — extract value from first match, then strip all occurrences
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

    # Strip structural verb fragments left after rubrique/filter extraction
    source = re.sub(r"\bparlant\b|\bprovenant\b", "", source, flags=re.IGNORECASE)

    source = re.sub(r"\s+", " ", source).strip()
    return source, filtres


def decouper_expression_logique(source: str) -> tuple[list[str], list[str]]:
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
    value = value.strip(" ,\"'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if not unicodedata.combining(c))


def supprimer_prefixe_theme(theme: str) -> str:
    theme = nettoyer_valeur_extraite(theme)
    # Strip "les mots X / le mot X" before article strip so "les" isn't consumed alone
    theme = re.sub(r"^(?:les?\s+mots?\b|les?\s+termes?\b)\s*", "", theme, flags=re.IGNORECASE)
    theme = re.sub(
        r"^(des|du|de la|de l['']|d['']|de|les|la|le|l[''])\s*",
        "",
        theme,
        flags=re.IGNORECASE,
    )
    # Strip any inline "le/les mot(s)/terme(s)" phrase (e.g. "possédant le mot France")
    theme = re.sub(r"\bles?\s+mots?\s+|\bles?\s+termes?\s+", " ", theme, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", theme).strip()


def est_partie_temporelle(partie: str) -> bool:
    partie = nettoyer_valeur_extraite(partie.lower())
    if not partie:
        return False

    if PATTERNS["between_dmy"].search(partie) or PATTERNS["between_my"].search(partie) or PATTERNS["between_y"].search(partie):
        return True

    if PATTERNS["dmy_numero"].search(partie) or PATTERNS["dmy_text"].search(partie):
        return True

    if PATTERNS["my_numero"].search(partie) or PATTERNS["my_text"].search(partie):
        return True

    if PATTERNS["y"].search(partie):
        mots_temporels = ("mois", "annee", "an", "apres", "avant", "depuis", "partir", "date", "publie")
        if any(mot in partie for mot in mots_temporels):
            return True

    return False


def extraire_theme_depuis_part_v2(partie: str, dernier_type: str) -> str | None:
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
        match_simple = re.search(r"\b(?:des|de l['’']|de la|de|du|d['’'])\s*(.+)$", partie, re.IGNORECASE)
        if match_simple is not None:
            theme = supprimer_prefixe_theme(match_simple.group(1))
            return theme if theme else None

    return nettoyer_valeur_extraite(partie)


def extraire_theme_negatif_depuis_part(partie: str) -> str | None:
    partie = partie.strip()
    if not partie:
        return None

    match = NEGATIVE_THEME_PATTERN.search(partie)
    if match is None:
        return None

    theme = supprimer_prefixe_theme(match.group("theme"))
    return theme if theme else None


def nettoyer_titre_suite(partie: str) -> str:
    partie = nettoyer_valeur_extraite(partie)
    partie = re.sub(r"^(?:et|ou)\s+", "", partie, flags=re.IGNORECASE)
    partie = re.sub(r"^(?:le|les)\s+(?:mot|mots|terme|termes)\s*", "", partie, flags=re.IGNORECASE)
    partie = nettoyer_valeur_extraite(partie)
    return partie


def nettoyer_theme_suite(partie: str) -> str:
    partie = nettoyer_valeur_extraite(partie)
    partie = re.sub(r"^(?:et|ou)\s+", "", partie, flags=re.IGNORECASE)
    partie = supprimer_prefixe_theme(partie)
    return nettoyer_valeur_extraite(partie)


def traiter_filtres_structurels(
    parts: list[str],
    operateurs: list[str],
) -> tuple[list[str], list[str], list[str], list[str], bool | None]:
    """Returns (parts_restants, title_keywords, themes, themes_exclus, image)"""
    parts_restantes: list[str] = []
    themes: list[str] = []
    themes_exclus: list[str] = []
    titres: list[str] = []
    image: bool | None = None
    dernier_type = None
    for i, part in enumerate(parts):
        part_courante = part.strip()
        if not part_courante:
            continue

        operateur_precedent = operateurs[i - 1] if i > 0 and i - 1 < len(operateurs) else None
        est_negatif = operateur_precedent in {"sans", "mais pas", "non pas", "et non pas"}

        if est_partie_temporelle(part_courante):
            parts_restantes.append(part_courante)
            dernier_type = None
            continue

        title_match = TITLE_CONTAINS_PATTERN.search(part_courante)
        if title_match is not None:
            dernier_type = "title"
            titre = nettoyer_valeur_extraite(title_match.group("value"))
            if titre:
                titres.append(titre)
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
                themes.append(theme)
                dernier_type = "theme"
            continue

        if dernier_type == "title" and operateur_precedent in {"et", "ou"}:
            titre_suite = nettoyer_titre_suite(part_courante)
            if titre_suite:
                titres.append(titre_suite)
                continue
            dernier_type = None

        if dernier_type == "theme" and operateur_precedent in {"et", "ou"}:
            theme_suite = nettoyer_theme_suite(part_courante)
            if theme_suite:
                themes.append(theme_suite)
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

    return parts_restantes, titres, themes, themes_exclus, image


def extraire_keywords_partie(partie: str, anti_list: list) -> list[str]:
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
    themes: list[str],
    themes_exclus: list[str],
    anti_list: list,
    upper_key_word: list[str],
) -> tuple[list[str], list[str]]:
    """Returns (key_word, key_word_exclu)"""
    keywords: list[str] = []
    exclu_keywords: list[str] = []

    # for partie in parts_restants:
    #     keywords.extend(extraire_keywords_partie(partie, anti_list))

    for theme in themes:
        keywords.extend(extraire_keywords_partie(theme, anti_list))

    for theme in themes_exclus:
        exclu_keywords.extend(extraire_keywords_partie(theme, anti_list))

    # title_keywords: NOT merged into key_word — moteur filters separately via _filtrer_titre

    key_word: list[str] = []
    deja_vus: set[str] = set()
    for word in keywords:
        if word not in deja_vus:
            key_word.append(word)
            deja_vus.add(word)
    for word in upper_key_word:
        mot_normalise = word.lower()
        if len(mot_normalise) <= 1 or mot_normalise in deja_vus:
            continue
        key_word.append(mot_normalise)
        deja_vus.add(mot_normalise)

    key_word_exclu: list[str] = []
    exclu_vus: set[str] = set()
    for word in exclu_keywords:
        if word not in exclu_vus:
            key_word_exclu.append(word)
            exclu_vus.add(word)

    return key_word, key_word_exclu


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
        result = pipeline_traitement_requete(request_norm, tf_idf_dict, anti_list, upper_kw)
        if any(result.values()):
            print(f"\n{request.strip()}")
            for k, v in result.items():
                if v:
                    print(f"{k}: {v}")
                    # print(f"  {k:<16}: {v}")
