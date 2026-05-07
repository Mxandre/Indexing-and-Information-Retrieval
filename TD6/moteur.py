import sys
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from datetime import datetime

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir / "TD5"))

from traitement_requete import (
    normaliser_texte,
    pipeline_traitement_requete,
    PICKLE_TF_IDF_FILE,
    PICKLE_ANTI_LIST,
    load_lemmatisatioin_file,
)

INDEX_INVERSE_FULL_FILE = parent_dir / "TD3" / "inverse_index_full.txt"
CORPUS_FILE = parent_dir / "TD3" / "corpus_filtre.xml"


def charger_index(filepath: Path) -> dict:
    """
    Load the inverse index from a text file.

    Format per line: lemma\tentry, entry, ...
    where each entry is doc_id.zone: freq (e.g., 273.titre: 1, 273.texte: 3)

    Returns: {lemma: {doc_id: {zone: freq}}}
    - All keys are strings
    - freq values are integers
    - Returns {} if file is missing
    """
    if not filepath.exists():
        print(f"Warning: Index file not found at {filepath}")
        return {}

    index = {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue

                try:
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue

                    lemma = parts[0].strip()
                    if not lemma:
                        continue

                    entries_str = parts[1].strip()
                    entries = [e.strip() for e in entries_str.split(",")]

                    doc_dict = defaultdict(dict)

                    for entry in entries:
                        if not entry:
                            continue

                        # Parse entry: doc_id.zone: freq
                        match = re.match(r"^(\d+)\.(\w+):\s*(\d+)$", entry)
                        if not match:
                            continue

                        doc_id = match.group(1)
                        zone = match.group(2)
                        freq = int(match.group(3))

                        doc_dict[doc_id][zone] = freq

                    if doc_dict:
                        index[lemma] = dict(doc_dict)

                except Exception:
                    # Skip malformed lines
                    continue

    except Exception as e:
        print(f"Error loading index from {filepath}: {e}")
        return {}

    return index


def charger_corpus(filepath: Path) -> dict:
    """
    Load the corpus from an XML file.

    XML structure:
    <corpus>
        <document>
            <bulletin>273</bulletin>
            <titre>...</titre>
            <date>11/09/2012</date>
            <rubrique>...</rubrique>
            <texte>...</texte>
            ...
        </document>
        ...
    </corpus>

    Returns: {bulletin_id: {"titre": ..., "date": ..., "rubrique": ..., "texte": ...}}
    - Keys are document bulletin IDs (strings)
    - Fields default to "" if missing or None
    - Returns {} if file is missing
    """
    if not filepath.exists():
        print(f"Warning: Corpus file not found at {filepath}")
        return {}

    corpus = {}

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for document in root.findall("document"):
            # Extract bulletin ID
            bulletin_elem = document.find("bulletin")
            if bulletin_elem is None or not bulletin_elem.text:
                continue

            bulletin_id = bulletin_elem.text.strip()
            if not bulletin_id:
                continue

            # Extract fields
            titre_elem = document.find("titre")
            titre = titre_elem.text.strip() if titre_elem is not None and titre_elem.text else ""

            date_elem = document.find("date")
            date = date_elem.text.strip() if date_elem is not None and date_elem.text else ""

            rubrique_elem = document.find("rubrique")
            rubrique = rubrique_elem.text.strip() if rubrique_elem is not None and rubrique_elem.text else ""

            texte_elem = document.find("texte")
            texte = texte_elem.text.strip() if texte_elem is not None and texte_elem.text else ""

            corpus[bulletin_id] = {
                "titre": titre,
                "date": date,
                "rubrique": rubrique,
                "texte": texte,
            }

    except Exception as e:
        print(f"Error loading corpus from {filepath}: {e}")
        return {}

    return corpus
