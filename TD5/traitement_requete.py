import csv
import heapq
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

try:
    import spacy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    spacy = None

KEYWORD_GAP_THRESHOLD = 0.35
KEYWORD_MIN_SCORE = 1.5
MAX_KEYWORDS = 5

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
DE_VARIANTS_PATTERN = r"(?:de|du|des|de la|de l[’']|d[’'])"

# Capture la rubrique jusqu'à une conjonction logique ou la fin.
# Ex: "rubrique au coeur des régions et ..." -> "au coeur des régions"
RUBRIQUE_PATTERN = re.compile(
    r"\brubrique\s+(?P<rubrique>.+?)(?=$|\b(?:et|ou|sans|mais\s+pas|non\s+pas)\b)",
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
REQUEST_TYPE_PATTERN = re.compile(r"\b(articles?|rubriques?|bulletins?)\b", re.IGNORECASE)
TITLE_CONTAINS_PATTERN = re.compile(
    r"\bdont\s+le\s+titre\s+(?:contient|evoque)\s+(?:le\s+mot|les\s+mots|le\s+terme)?\s*\"?(?P<value>[^\"]+?)\"?(?:$|\b(?:et|ou|mais\s+pas|sans)\b)",
    re.IGNORECASE,
)
IMAGE_PATTERN = re.compile(r"\bavec\s+des?\s+images?\b|\bavec\s+image\b|\bcontenant\s+une?\s+image\b|\bqui\s+ont\s+des?\s+images?\b", re.IGNORECASE)
WITHOUT_IMAGE_PATTERN = re.compile(r"\bsans\s+image\b|\bsans\s+images\b", re.IGNORECASE)
THEME_TRIGGER_PATTERN = re.compile(
    rf"\b(?:parl(?:e|ent|ant|er)(?:\s+{DE_VARIANTS_PATTERN})?|trait(?:e|ant|er)\s+{DE_VARIANTS_PATTERN}|sur|a\s+propos\s+{DE_VARIANTS_PATTERN}|evoqu(?:e|ent|ant|er)|mentionn(?:e|ent|ant|er)|port(?:e|ent|ant|er)\s+sur|li(?:e|es)\s+a|concern(?:e|ent))\b\s*(?P<theme>.+)",
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
    # Fallback: charge depuis le fichier source si le pickle n'existe pas.
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
    # Trie décroissant pour favoriser les rubriques multi-mots
    _RUBRIQUES.sort(key=len, reverse=True)
    return _RUBRIQUES

def load_lemmatisatioin_file():
    """
    Construit et sérialise:
    - `lemma_dict.pkl` (inflection -> lemma)
    - `tf-idf.pkl` (lemma -> score)
    - `anti_list.pkl` (stopwords)
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
    tree = ET.parse(file_xml)
    root = tree.getroot()
    rubriques = set()

    for doc in root.findall("document"):
        elem = doc.find("rubrique")
        if elem is not None and elem.text:
            rubriques.add(elem.text.strip())
    return list(rubriques)
  

def normaliser_texte(source: str, key_word_traite = False) -> str:
    source = source.replace("?", " ")
    source = re.sub(r"\s+", " ", source)
    
    words = re.findall(r"\b\w+\b", source)
    key_word = []

    if key_word_traite:
        for word in words :
            if any(c.isalpha() for c in word) and word.isupper():
                key_word.append(word)
    source = source.lower()
    if key_word_traite:
        return source.strip(), key_word
    else:
        return source.strip()


def pipeline_traitement_requete(source: str, metadonnees: dict, tf_idf_dict, anti_list, upper_key_word) -> dict:
    metadonnees = traiter_type_requete(source, metadonnees)
    metadonnees = traiter_metadonnees(source, metadonnees)
    source_sans_prefixe = supprimer_prefixe_avant_articles(source)
    metadonnees["source_nettoyee"] = source_sans_prefixe
    metadonnees = traiter_op_logique(source_sans_prefixe, metadonnees)
    metadonnees = traiter_filtres_structurels(metadonnees)
    metadonnees = traiter_mots_cles(source_sans_prefixe, metadonnees, tf_idf_dict, anti_list, upper_key_word)
    return metadonnees


def traiter_metadonnees(source: str, metadonnees: defaultdict) -> dict:
    """
    Traite les metadonnees de date et de rubrique.
    """
    source_normalisee = normaliser_texte(source)

    between_dmy = PATTERNS["between_dmy"].search(source_normalisee)
    if between_dmy is not None:
        metadonnees["date"] = {
            "type": "between_dmy",
            "start": between_dmy.group("start"),
            "end": between_dmy.group("end"),
        }
    else:
        between_my = PATTERNS["between_my"].search(source_normalisee)
        if between_my is not None:
            metadonnees["date"] = {
                "type": "between_my",
                "start": between_my.group("start"),
                "end": between_my.group("end"),
            }
        else:
            between_y = PATTERNS["between_y"].search(source_normalisee)
            if between_y is not None:
                metadonnees["date"] = {
                    "type": "between_y",
                    "start": between_y.group("start"),
                    "end": between_y.group("end"),
                }
            else:
                dmy_numero = PATTERNS["dmy_numero"].search(source_normalisee)
                dmy_texte = PATTERNS["dmy_text"].search(source_normalisee)
                if dmy_numero is not None or dmy_texte is not None:
                    date_match = dmy_numero if dmy_numero else dmy_texte
                    metadonnees["date"] = {
                        "type": "dmy",
                        "value": date_match.group(0),
                    }
                else:
                    my_numero = PATTERNS["my_numero"].search(source_normalisee)
                    my_texte = PATTERNS["my_text"].search(source_normalisee)
                    if my_numero is not None or my_texte is not None:
                        date_match = my_numero if my_numero else my_texte
                        metadonnees["date"] = {
                            "type": "my",
                            "value": date_match.group(0),
                        }
                    else:
                        y_pattern = PATTERNS["y"].search(source_normalisee)
                        if y_pattern is not None:
                            metadonnees["date"] = {
                                "type": "y",
                                "value": y_pattern.group(0),
                            }

    rubrique_match = PATTERNS["rubrique"].search(source_normalisee)
    if rubrique_match is not None:
        rubrique_raw = nettoyer_valeur_extraite(rubrique_match.group("rubrique")).lower()
        if rubrique_raw:
            rubriques_connues = get_rubriques_corpus()
            rubrique_finale = None
            for r in rubriques_connues:
                if rubrique_raw.startswith(r):
                    rubrique_finale = r
                    break
            metadonnees["rubrique"] = rubrique_finale or rubrique_raw

    return metadonnees


def traiter_type_requete(source: str, metadonnees: dict) -> dict:
    match = REQUEST_TYPE_PATTERN.search(source)
    if match is not None:
        metadonnees["request_type"] = match.group(1).lower()
    return metadonnees


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


def traiter_op_logique(source: str, metadonnees: dict) -> dict:
    source_normalisee = normaliser_texte(source)
    parties, operateurs = decouper_expression_logique(source_normalisee)

    parts_bruts = parties if parties else [source_normalisee]
    metadonnees["parts_bruts"] = parts_bruts
    if operateurs:
        metadonnees["operateurs"] = operateurs
    return metadonnees


def nettoyer_valeur_extraite(value: str) -> str:
    value = value.strip(" ,\"'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def supprimer_prefixe_theme(theme: str) -> str:
    theme = nettoyer_valeur_extraite(theme)
    theme = re.sub(
        r"^(des|du|de la|de l[’']|d[’']|de|les|la|le|l[’'])\s*",
        "",
        theme,
        flags=re.IGNORECASE,
    )
    return theme.strip()


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



def extraire_theme_depuis_part_v2(partie: str) -> str | None:
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

    match_simple = re.search(r"\b(?:des|de|du|de la|de l[’']|d[’'])\s*(.+)$", partie, re.IGNORECASE)
    if match_simple is not None:
        theme = supprimer_prefixe_theme(match_simple.group(1))
        return theme if theme else None

    return nettoyer_valeur_extraite(partie)


def traiter_filtres_structurels(metadonnees: dict) -> dict:
    parts = metadonnees.get("parts_bruts", [])
    operateurs = metadonnees.get("operateurs", [])
    parts_restantes = []
    themes = []
    themes_exclus = []
    titres = []

    for i, part in enumerate(parts):
        part_courante = part.strip()
        if not part_courante:
            continue

        operateur_precedent = operateurs[i - 1] if i > 0 and i - 1 < len(operateurs) else None
        est_negatif = operateur_precedent in {"sans", "mais pas", "non pas", "et non pas"}

        if est_partie_temporelle(part_courante):
            parts_restantes.append(part_courante)
            continue

        title_match = TITLE_CONTAINS_PATTERN.search(part_courante)
        if title_match is not None:
            titre = nettoyer_valeur_extraite(title_match.group("value"))
            if titre:
                titres.append(titre)
            continue

        if WITHOUT_IMAGE_PATTERN.search(part_courante):
            metadonnees["image"] = False
            continue

        if part_courante.lower() in {"image", "images"} and est_negatif:
            metadonnees["image"] = False
            continue

        if IMAGE_PATTERN.search(part_courante):
            metadonnees["image"] = True
            continue

        if part_courante.lower() in {"image", "images"}:
            metadonnees["image"] = True
            continue

        theme = extraire_theme_depuis_part_v2(part_courante)

        if theme and (theme != part_courante or est_negatif):
            if est_negatif:
                themes_exclus.append(theme)
            else:
                themes.append(theme)
        else:
            parts_restantes.append(part_courante)

    if titres:
        metadonnees["title_keywords"] = titres
    if themes:
        metadonnees["themes"] = themes
    if themes_exclus:
        metadonnees["themes_exclus"] = themes_exclus

    metadonnees["parts_themes_restants"] = parts_restantes
    return metadonnees


def selectionner_keywords_dynamiques(candidats: list[tuple[str, float]]) -> list[str]:
    if not candidats:
        return []

    candidats_tries = sorted(candidats, key=lambda x: x[1], reverse=True)
    meilleur_score = candidats_tries[0][1]
    if meilleur_score < KEYWORD_MIN_SCORE:
        # Fallback: sur des requêtes "courantes", les tf-idf peuvent être faibles.
        # On garde au moins le meilleur candidat pour éviter de renvoyer une requête vide.
        return [candidats_tries[0][0]]

    selection = [candidats_tries[0][0]]

    for i in range(1, min(len(candidats_tries), MAX_KEYWORDS)):
        score_precedent = candidats_tries[i - 1][1]
        score_courant = candidats_tries[i][1]
        if score_precedent - score_courant > KEYWORD_GAP_THRESHOLD:
            break
        selection.append(candidats_tries[i][0])

    return selection


def extraire_keywords_partie(partie: str, tf_idf_dict, anti_list) -> list[str]:
    heap: list[tuple[str, float]] = []
    meilleurs_scores: dict[str, float] = {}
    nlp = get_nlp()

    if nlp is not None:
        doc = nlp(partie)
        tokens = []
        for token in doc:
            if not token.is_alpha:
                continue
            tokens.append(token.lemma_.lower())
    else:
        lemma_dict = get_lemma_dict()
        tokens = []
        for w in re.findall(r"[A-Za-zÀ-ÿ]+", partie.lower()):
            tokens.append(lemma_dict.get(w, w))

    for word_lemma in tokens:
        if word_lemma in anti_list:
            continue
        tf_idf = tf_idf_dict.get(word_lemma)
        if tf_idf is None:
            continue
        ancien_score = meilleurs_scores.get(word_lemma)
        if ancien_score is None or tf_idf > ancien_score:
            meilleurs_scores[word_lemma] = tf_idf

    for mot, score in meilleurs_scores.items():
        heapq.heappush(heap, (mot, score))

    candidats = heapq.nlargest(len(heap), heap, key=lambda x: x[1])
    return selectionner_keywords_dynamiques(candidats)


def traiter_mots_cles(source: str, metadonnees: dict, tf_idf_dict, anti_list, upper_key_word) -> dict:
    parties = metadonnees.get("parts_themes_restants", metadonnees.get("parts_bruts", [source]))
    keywords = []

    for partie in parties:
        keywords.extend(extraire_keywords_partie(partie, tf_idf_dict, anti_list))

    for theme in metadonnees.get("themes", []):
        keywords.extend(extraire_keywords_partie(theme, tf_idf_dict, anti_list))

    for theme in metadonnees.get("themes_exclus", []):
        keywords.extend(extraire_keywords_partie(theme, tf_idf_dict, anti_list))

    for titre in metadonnees.get("title_keywords", []):
        mot_titre = nettoyer_valeur_extraite(titre).lower()
        if mot_titre:
            keywords.append(mot_titre)

    deja_vus = set()
    for word in keywords:
        if word not in deja_vus:
            metadonnees["key_word"].append(word)
            deja_vus.add(word)

    for word in upper_key_word:
        mot_normalise = word.lower()
        if mot_normalise not in deja_vus:
            metadonnees["key_word"].append(mot_normalise)
            deja_vus.add(mot_normalise)
    return metadonnees


def extraire_request(file_path: Path) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        contenu = f.read()
    contenu = re.sub(r"\s+", " ", contenu)
    morceaux = contenu.split("?")
    requests = []
    for morceau in morceaux:
        petit_morceaux = morceau.split(".")
        for petit_morceau in petit_morceaux:
            req = petit_morceau.strip()
            req = re.sub(r"^[^\wA-Za-zÀ-ÿ]+", "", req)
            if req:
                requests.append(req)
    return requests


if __name__ == "__main__":
    
    file_name = Path(__file__).parent / "requete.txt"
    load_lemmatisatioin_file()
    rubirque_file = get_rubrique(RUBRIQUE_FILE)

    with open(file_name, "r", encoding="utf-8") as f:
        requests = f.readlines()
    with open(PICKLE_TF_IDF_FILE, 'rb') as file:
        tf_idf_dict = pickle.load(file)
    with open(PICKLE_ANTI_LIST, "rb") as file :
        anti_list = pickle.load(file)
    
    for request in requests:
        metadonnes = defaultdict(list)
        for rubrique in rubirque_file :
            if rubrique in request :
                metadonnes["rubrique"].append(rubrique)
        request_normalisee, upper_key_word = normaliser_texte(request, key_word_traite=True)
        metadonnes = pipeline_traitement_requete(request_normalisee, metadonnes, tf_idf_dict, anti_list,upper_key_word)
        
        if len(metadonnes.keys()) >= 1:
            print(request.strip())
            # print(request.strip().encode("ascii", errors="backslashreplace").decode("ascii"))
            print("the result obtained is", dict(metadonnes))
