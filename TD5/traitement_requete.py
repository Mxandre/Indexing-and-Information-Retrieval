import re
import unicodedata
from pathlib import Path
from collections import defaultdict


MONTH = r"janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre"
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

RUBRIQUE_PATTERN = re.compile(r"\brubrique\s+(?P<rubrique>[A-Za-zÀ-ÿ\-]+)\b", re.IGNORECASE)
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


def normaliser_texte(source: str) -> str:
    source = unicodedata.normalize("NFKD", source)
    source = "".join(ch for ch in source if not unicodedata.combining(ch))
    source = source.replace("?", " ")
    source = re.sub(r"\s+", " ", source)
    return source.strip()


def pipeline_traitement_requete(source: str, metadonnees: dict) -> dict:
    metadonnees = traiter_metadonnees(source, metadonnees)
    metadonnees = traiter_op_logique(source, metadonnees)
    metadonnees = traiter_mots_cles(source, metadonnees)
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

    rubrique = PATTERNS["rubrique"].search(source_normalisee)
    if rubrique:
        metadonnees["rubrique"] = rubrique.group("rubrique")

    return metadonnees


def traiter_op_logique(source: str, metadonnees: dict) -> dict:
    op_match = PATTERNS["op"].search(normaliser_texte(source))
    if op_match:
        metadonnees["operateur"] = op_match.group("op")
        metadonnees["part1"] = op_match.group("part1").strip()
        metadonnees["part2"] = op_match.group("part2").strip()
    return metadonnees


def traiter_mots_cles(source: str, metadonnees: dict) -> dict:
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

    with open(file_name, "r", encoding="utf-8") as f:
        requests = f.readlines()

    for request in requests:
        metadonnes = defaultdict(list)
        metadonnes = traiter_metadonnees(request, metadonnes)
        if len(metadonnes.keys()) >= 1:
            print(request.strip().encode("ascii", errors="backslashreplace").decode("ascii"))
            print("the result obtained is", dict(metadonnes))
