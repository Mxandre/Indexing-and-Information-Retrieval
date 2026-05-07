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


def evaluer_metadonnees(metadonnees: dict, index_inverse: dict) -> dict[str, float]:
    """
    Score documents against metadata constraints (keywords, rubrique, date).

    Returns {doc_id: score} where score is the sum of keyword contributions.
    For rubrique/date-only results, score is 0.0.
    Returns {} if all filters are absent.

    Scoring for keywords:
      score contribution per keyword per doc = freq_titre * 3 + freq_texte * 1
    Intersection across ALL keywords is required.

    Rubrique and date are AND-filtered on top of keyword results.
    """
    keywords = metadonnees.get("key_word", [])
    rubrique = metadonnees.get("rubrique", None)
    date_info = metadonnees.get("date", None)

    # Empty case
    if not keywords and rubrique is None and date_info is None:
        return {}

    # ------------------------------------------------------------------ #
    # Step 1: keyword scoring with intersection across all keywords        #
    # ------------------------------------------------------------------ #
    keyword_scores: dict[str, float] | None = None  # None = "not filtered yet"

    if keywords:
        for keyword in keywords:
            kw_docs = index_inverse.get(keyword, {})
            # Compute per-doc score for this keyword
            kw_scores: dict[str, float] = {}
            for doc_id, zones in kw_docs.items():
                freq_titre = zones.get("titre", 0)
                freq_texte = zones.get("texte", 0)
                kw_scores[doc_id] = freq_titre * 3 + freq_texte * 1

            if keyword_scores is None:
                keyword_scores = kw_scores
            else:
                # Intersection: keep only docs present for ALL keywords, sum scores
                new_scores: dict[str, float] = {}
                for doc_id, score in keyword_scores.items():
                    if doc_id in kw_scores:
                        new_scores[doc_id] = score + kw_scores[doc_id]
                keyword_scores = new_scores

        if keyword_scores is None:
            keyword_scores = {}

    # ------------------------------------------------------------------ #
    # Step 2: rubrique filter                                              #
    # ------------------------------------------------------------------ #
    rubrique_docs: set[str] | None = None
    if rubrique is not None:
        rubrique_lower = rubrique.lower()
        rub_entries = index_inverse.get(rubrique_lower, {})
        rubrique_docs = set(rub_entries.keys())

    # ------------------------------------------------------------------ #
    # Step 3: date filter                                                  #
    # ------------------------------------------------------------------ #
    date_docs: set[str] | None = None
    if date_info is not None:
        dtype = date_info.get("type", "")

        if dtype in ("dmy", "my", "y"):
            # Single value lookup
            val = date_info.get("value", "")
            date_docs = set(index_inverse.get(val, {}).keys())

        elif dtype == "between_dmy":
            start_str = date_info.get("start", "")
            end_str = date_info.get("end", "")
            try:
                start_dt = datetime.strptime(start_str, "%d/%m/%Y")
                end_dt = datetime.strptime(end_str, "%d/%m/%Y")
            except ValueError:
                start_dt = end_dt = None

            if start_dt is None or end_dt is None:
                print(f"Avertissement : plage de dates non analysable ({date_info})")

            date_docs = set()
            for key, docs in index_inverse.items():
                try:
                    key_dt = datetime.strptime(key, "%d/%m/%Y")
                    if start_dt is not None and end_dt is not None:
                        if start_dt <= key_dt <= end_dt:
                            date_docs.update(docs.keys())
                except ValueError:
                    continue

        elif dtype == "between_my":
            start_str = date_info.get("start", "")
            end_str = date_info.get("end", "")
            # Format: "mm/yyyy"
            try:
                start_dt = datetime.strptime(start_str, "%m/%Y")
                end_dt = datetime.strptime(end_str, "%m/%Y")
            except ValueError:
                start_dt = end_dt = None

            if start_dt is None or end_dt is None:
                print(f"Avertissement : plage de dates non analysable ({date_info})")

            date_docs = set()
            for key, docs in index_inverse.items():
                try:
                    key_dt = datetime.strptime(key, "%m/%Y")
                    if start_dt is not None and end_dt is not None:
                        if start_dt <= key_dt <= end_dt:
                            date_docs.update(docs.keys())
                except ValueError:
                    continue

        elif dtype == "between_y":
            start_str = date_info.get("start", "")
            end_str = date_info.get("end", "")
            date_docs = set()
            for key, docs in index_inverse.items():
                # Match keys that look like a 4-digit year
                if re.fullmatch(r"\d{4}", key):
                    try:
                        if int(start_str) <= int(key) <= int(end_str):
                            date_docs.update(docs.keys())
                    except ValueError:
                        pass

    # ------------------------------------------------------------------ #
    # Step 4: combine results                                              #
    # ------------------------------------------------------------------ #

    if keyword_scores is not None:
        # Start from keyword results, intersect with rubrique and/or date
        result = keyword_scores
        if rubrique_docs is not None:
            result = {doc_id: score for doc_id, score in result.items() if doc_id in rubrique_docs}
        if date_docs is not None:
            result = {doc_id: score for doc_id, score in result.items() if doc_id in date_docs}
        return result

    else:
        # No keywords — combine rubrique and/or date sets with score 0.0
        combined: set[str] | None = None
        if rubrique_docs is not None:
            combined = rubrique_docs
        if date_docs is not None:
            if combined is None:
                combined = date_docs
            else:
                combined = combined & date_docs
        if combined is None:
            return {}
        return {doc_id: 0.0 for doc_id in combined}


def evaluer_requete_recursive(
    requete_texte: str,
    tf_idf_dict: dict,
    anti_list: list,
    index_inverse: dict,
) -> set:
    """
    Evaluate a boolean query recursively.

    Recursively splits compound queries (ET / OU / SANS) into sub-queries,
    evaluates each leaf via evaluer_metadonnees, and combines results with
    set operations.

    Returns a set of doc_id strings.
    """
    # Step 1: normalise the raw query text
    req_norm, upper_kw = normaliser_texte(requete_texte, key_word_traite=True)

    # Step 2: parse the query into metadata / operator structure
    meta = pipeline_traitement_requete(req_norm, defaultdict(list), tf_idf_dict, anti_list, upper_kw)

    # Step 3: compound query — recurse on both parts
    if "operateur" in meta and "part1" in meta and "part2" in meta:
        res1 = evaluer_requete_recursive(meta["part1"], tf_idf_dict, anti_list, index_inverse)
        res2 = evaluer_requete_recursive(meta["part2"], tf_idf_dict, anti_list, index_inverse)

        operateur = meta["operateur"]
        if operateur == "et":
            return res1 & res2
        elif operateur == "ou":
            return res1 | res2
        else:
            # "sans", "mais pas", "non pas", "et non pas" → difference
            return res1 - res2

    # Step 4: leaf node — score documents and return as a set of doc_ids
    scored = evaluer_metadonnees(meta, index_inverse)
    return set(scored.keys())


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


def generer_snippets(texte: str, keywords: list, window: int = 40) -> list:
    """
    Find ALL occurrences of each keyword in texte and return contextual snippets.

    Returns a list of snippet strings with keyword occurrences highlighted in [brackets].
    Overlapping windows are merged. A fallback of the first 120 chars is returned if
    no occurrences are found.
    """
    if not keywords or not texte:
        return [texte[:120] + "..."] if texte else []

    texte_lower = texte.lower()

    # Step 1: collect all (start, end, kw_found) tuples for each keyword occurrence
    occurrences = []
    for kw in keywords:
        kw_lower = kw.lower()
        for m in re.finditer(re.escape(kw_lower), texte_lower):
            idx = m.start()
            start = max(0, idx - window)
            end = min(len(texte), idx + len(kw) + window)
            occurrences.append((start, end))

    if not occurrences:
        return [texte[:120] + "..."]

    # Step 2: sort by start position
    occurrences.sort(key=lambda x: x[0])

    # Step 3: merge overlapping windows
    merged = []
    cur_start, cur_end = occurrences[0]
    for start, end in occurrences[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    # Step 4: build snippets with bracket highlighting
    snippets = []
    for win_start, win_end in merged:
        chunk = texte[win_start:win_end]

        # Highlight each keyword (case-insensitive, preserve original case)
        # Collect all match spans on the ORIGINAL chunk first to avoid nested brackets
        all_matches = []
        for kw in set(keywords):  # deduplicate keywords
            for m in re.finditer(re.escape(kw), chunk, re.IGNORECASE):
                all_matches.append((m.start(), m.end(), m.group(0)))
        all_matches.sort(key=lambda x: x[0])
        result_parts = []
        pos = 0
        for start, end, found in all_matches:
            if start < pos:
                continue  # skip overlapping match
            result_parts.append(chunk[pos:start])
            result_parts.append(f"[{found}]")
            pos = end
        result_parts.append(chunk[pos:])
        chunk = "".join(result_parts)

        # Add ellipsis prefix/suffix
        prefix = "..." if win_start > 0 else ""
        suffix = "..." if win_end < len(texte) else ""
        snippets.append(prefix + chunk + suffix)

    return snippets


def afficher_resultats(doc_ids: set, corpus_data: dict, keywords: list, mode: str) -> None:
    """
    Display formatted search results to stdout.

    Parameters:
    - doc_ids: set of document IDs from evaluer_requete_recursive
    - corpus_data: {bulletin_id: {titre, date, rubrique, texte}}
    - keywords: list of keywords for snippet generation
    - mode: "classe" or "booleen"
    """
    print(f"\n--- {len(doc_ids)} résultat(s) trouvé(s) ---\n")

    if not doc_ids:
        print("Aucun document trouvé.")
        return

    # Build list of (doc_id, data) tuples for sorting
    docs = []
    for doc_id in doc_ids:
        data = corpus_data.get(doc_id, {})
        docs.append((doc_id, data))

    if mode == "classe":
        # Sort by keyword frequency in texte + titre (descending)
        def keyword_freq(item):
            _, data = item
            combined = (data.get("texte", "") + " " + data.get("titre", "")).lower()
            return sum(combined.count(kw.lower()) for kw in keywords)

        docs.sort(key=keyword_freq, reverse=True)

    else:  # mode == "booleen"
        # Sort by date descending (dd/mm/yyyy), unparseable dates go last
        def sort_key(item):
            doc_id, data = item
            date_str = data.get("date", "")
            try:
                ts = datetime.strptime(date_str, "%d/%m/%Y").timestamp()
                return (0, -ts)  # (0, negative timestamp) → more recent = smaller = sorts first
            except (ValueError, TypeError):
                return (1, 0)    # unparseable → sorts after all valid dates

        docs.sort(key=sort_key)

    # Display results
    for i, (doc_id, data) in enumerate(docs, start=1):
        titre = data.get("titre", "")
        date = data.get("date", "")
        rubrique = data.get("rubrique", "")
        texte = data.get("texte", "")

        print(f"[{i}] Doc #{doc_id} | Date: {date} | Rubrique: {rubrique}")
        print(f"    Titre : {titre}")

        # Generate snippets from the full text
        snips = generer_snippets(texte, keywords)
        if snips:
            print("    Extraits :")
            for snip in snips:
                print(f"      {snip}")

        if i < len(docs):
            print()


def lancer_moteur() -> None:
    """
    Main interactive search engine loop.
    """
    mode = "classe"
    print("================================================")
    print(f"    MOTEUR DE RECHERCHE — Mode: {mode}")
    print("================================================")
    print("Commandes : /mode booleen | /mode classe | /quitter")

    # Check/generate pickle files
    if not PICKLE_TF_IDF_FILE.exists() or not PICKLE_ANTI_LIST.exists():
        load_lemmatisatioin_file()

    # Load data
    with open(str(PICKLE_TF_IDF_FILE), "rb") as f:
        tf_idf_dict = pickle.load(f)
    with open(str(PICKLE_ANTI_LIST), "rb") as f:
        anti_list = pickle.load(f)

    index_inverse = charger_index(INDEX_INVERSE_FULL_FILE)
    corpus_data = charger_corpus(CORPUS_FILE)

    print("Moteur prêt !\n")

    while True:
        try:
            requete = input("> ").strip()
        except EOFError:
            print("\nAu revoir !")
            break

        if not requete:
            continue

        if requete == "/quitter":
            print("Au revoir !")
            break
        elif requete == "/mode booleen":
            mode = "booleen"
            print("Mode : booléen")
            continue
        elif requete == "/mode classe":
            mode = "classe"
            print("Mode : classé")
            continue

        # Evaluate query
        doc_ids = evaluer_requete_recursive(requete, tf_idf_dict, anti_list, index_inverse)

        # Extract keywords for snippet generation
        req_norm, upper_kw = normaliser_texte(requete, key_word_traite=True)
        meta = pipeline_traitement_requete(req_norm, defaultdict(list), tf_idf_dict, anti_list, upper_kw)
        keywords = meta.get("key_word", [])

        afficher_resultats(doc_ids, corpus_data, keywords, mode)


if __name__ == "__main__":
    lancer_moteur()
