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
    masquer_intervalles_temporels,
    LOGICAL_OP_PATTERN,
    extraire_filtres_globaux,
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
    title_keywords = metadonnees.get("title_keywords", [])
    rubrique = metadonnees.get("rubrique", None)
    date_info = metadonnees.get("date", None)

    # Empty case
    if not keywords and not title_keywords and rubrique is None and date_info is None:
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

    if not keywords and title_keywords:
        keyword_scores = None
        for kw in title_keywords:
            kw_docs = index_inverse.get(kw, {})
            kw_scores: dict[str, float] = {
                doc_id: zones.get("titre", 0) * 3
                for doc_id, zones in kw_docs.items()
                if zones.get("titre", 0) > 0
            }
            if keyword_scores is None:
                keyword_scores = kw_scores
            else:
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
        rub_entries = {
            doc_id: zones
            for doc_id, zones in index_inverse.get(rubrique_lower, {}).items()
            if "rubrique" in zones
        }
        rubrique_docs = set(rub_entries.keys())

    # ------------------------------------------------------------------ #
    # Step 3: date filter                                                  #
    # ------------------------------------------------------------------ #
    date_docs: set[str] | None = None
    if date_info is not None:
        dtype = date_info.get("type", "")

        if dtype == "dmy":
            # Exact date lookup (dd/mm/yyyy key exists in index)
            val = date_info.get("value", "")
            date_docs = set(index_inverse.get(val, {}).keys())

        elif dtype == "y":
            year_val = date_info["value"]  # e.g. "2012"
            date_docs = set()
            for key, docs in index_inverse.items():
                try:
                    dt = datetime.strptime(key, "%d/%m/%Y")
                    if str(dt.year) == year_val:
                        date_docs.update(docs.keys())
                except ValueError:
                    pass

        elif dtype == "my":
            my_str = date_info["value"]  # e.g. "09/2012" or "septembre 2012"
            date_docs = set()
            # Try parsing as m/yyyy first
            for fmt in ("%m/%Y", "%m-%Y", "%m %Y"):
                try:
                    target = datetime.strptime(my_str, fmt)
                    for key, docs in index_inverse.items():
                        try:
                            dt = datetime.strptime(key, "%d/%m/%Y")
                            if dt.month == target.month and dt.year == target.year:
                                date_docs.update(docs.keys())
                        except ValueError:
                            pass
                    break
                except ValueError:
                    continue

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
            try:
                start_y = int(start_str)
                end_y = int(end_str)
            except ValueError:
                start_y = end_y = None
            if start_y is not None:
                for key, docs in index_inverse.items():
                    try:
                        dt = datetime.strptime(key, "%d/%m/%Y")
                        if start_y <= dt.year <= end_y:
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


def _soustraire_exclus(docs: dict, key_word_exclu: list, index_inverse: dict) -> dict:
    if not key_word_exclu:
        return docs
    exclu_docs: set = set()
    for kw in key_word_exclu:
        exclu_docs.update(index_inverse.get(kw, {}).keys())
    return {doc_id: score for doc_id, score in docs.items() if doc_id not in exclu_docs}


def _filtrer_titre(docs: dict, title_keywords: list, index_inverse: dict) -> dict:
    if not title_keywords:
        return docs
    return {
        doc_id: score
        for doc_id, score in docs.items()
        if all(
            index_inverse.get(kw, {}).get(doc_id, {}).get("titre", 0) > 0
            for kw in title_keywords
        )
    }


def _appliquer_filtres_globaux(doc_ids: set, global_filters: dict, index_inverse: dict) -> set:
    rubrique = global_filters.get("rubrique")
    date_info = global_filters.get("date")
    if not rubrique and not date_info:
        return doc_ids
    fake_meta: dict = {}
    if rubrique:
        fake_meta["rubrique"] = rubrique
    if date_info:
        fake_meta["date"] = date_info
    filter_set = set(evaluer_metadonnees(fake_meta, index_inverse).keys())
    return doc_ids & filter_set


def _evaluer_recursive(
    source: str,
    tf_idf_dict: dict,
    anti_list: list,
    index_inverse: dict,
) -> set:
    req_norm = normaliser_texte(source)
    source_masque = masquer_intervalles_temporels(req_norm)
    op_match = LOGICAL_OP_PATTERN.search(source_masque)

    if op_match:
        left = req_norm[: op_match.start()].strip()
        right = req_norm[op_match.end() :].strip()
        op = op_match.group(0).lower().strip()
        res_left = _evaluer_recursive(left, tf_idf_dict, anti_list, index_inverse)
        res_right = _evaluer_recursive(right, tf_idf_dict, anti_list, index_inverse)
        if op == "et":
            return res_left & res_right
        elif op == "ou":
            return res_left | res_right
        else:  # sans, mais pas, non pas, et non pas
            return res_left - res_right

    # Leaf: process sub-expression through pipeline
    req_norm_kw, upper_kw = normaliser_texte(source, key_word_traite=True)
    meta = pipeline_traitement_requete(req_norm_kw, tf_idf_dict, anti_list, upper_kw)
    docs = evaluer_metadonnees(meta, index_inverse)
    docs = _soustraire_exclus(docs, meta.get("key_word_exclu", []), index_inverse)
    docs = _filtrer_titre(docs, meta.get("title_keywords", []), index_inverse)
    return set(docs.keys())


def evaluer_requete_recursive(
    requete_texte: str,
    tf_idf_dict: dict,
    anti_list: list,
    index_inverse: dict,
) -> set:
    """
    Evaluate a boolean/ranked query.

    Extracts global filters (rubrique, date, image) first, then evaluates the
    remaining expression with binary recursion on ET/OU/SANS.

    Returns a set of doc_id strings.
    """
    source_stripped, global_filters = extraire_filtres_globaux(requete_texte)
    doc_ids = _evaluer_recursive(source_stripped, tf_idf_dict, anti_list, index_inverse)
    return _appliquer_filtres_globaux(doc_ids, global_filters, index_inverse)


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

            if bulletin_id in corpus:
                existing = corpus[bulletin_id]
                if rubrique and rubrique not in existing["rubriques"]:
                    existing["rubriques"].append(rubrique)
                if texte:
                    existing["texte"] += " " + texte
            else:
                corpus[bulletin_id] = {
                    "titre": titre,
                    "date": date,
                    "rubriques": [rubrique] if rubrique else [],
                    "texte": texte,
                }

    except Exception as e:
        print(f"Error loading corpus from {filepath}: {e}")
        return {}

    return corpus


def charger_corpus_articles(filepath: Path) -> tuple[dict, dict]:
    """
    Returns (corpus_articles, bulletin_to_articles).
    corpus_articles      = {article_id: {article, bulletin, date, rubrique, titre, texte}}
    bulletin_to_articles = {bulletin_id: [article_id, ...]}  (insertion order)
    """
    if not filepath.exists():
        print(f"Warning: Corpus file not found at {filepath}")
        return {}, {}

    corpus_articles: dict = {}
    bulletin_to_articles: dict = {}

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for document in root.findall("document"):
            article_elem = document.find("article")
            if article_elem is None or not article_elem.text:
                continue
            article_id = article_elem.text.strip()
            if not article_id:
                continue

            bulletin_elem = document.find("bulletin")
            bulletin_id = bulletin_elem.text.strip() if bulletin_elem is not None and bulletin_elem.text else ""

            titre_elem = document.find("titre")
            titre = titre_elem.text.strip() if titre_elem is not None and titre_elem.text else ""

            date_elem = document.find("date")
            date = date_elem.text.strip() if date_elem is not None and date_elem.text else ""

            rubrique_elem = document.find("rubrique")
            rubrique = rubrique_elem.text.strip() if rubrique_elem is not None and rubrique_elem.text else ""

            texte_elem = document.find("texte")
            texte = texte_elem.text.strip() if texte_elem is not None and texte_elem.text else ""

            corpus_articles[article_id] = {
                "article": article_id,
                "bulletin": bulletin_id,
                "date": date,
                "rubrique": rubrique,
                "titre": titre,
                "texte": texte,
            }

            if bulletin_id:
                bulletin_to_articles.setdefault(bulletin_id, []).append(article_id)

    except Exception as e:
        print(f"Error loading corpus articles from {filepath}: {e}")
        return {}, {}

    return corpus_articles, bulletin_to_articles


def filtrer_articles_pertinents(
    bulletin_ids: set,
    keywords: list,
    title_keywords: list,
    rubrique_filter: str | None,
    corpus_articles: dict,
    bulletin_to_articles: dict,
) -> list[tuple[str, dict, float]]:
    """
    Resolve bulletin IDs to individual articles and score by keyword relevance.

    For each bulletin: score articles by keyword presence in titre (x3) and texte (x1).
    If rubrique_filter: prefer articles whose rubrique matches; fall back to all if none match.
    Keep articles with score > 0; if none, keep all selected (trust the index match).
    """
    results: list[tuple[str, dict, float]] = []
    all_kws = keywords + title_keywords

    for bulletin_id in bulletin_ids:
        article_ids = bulletin_to_articles.get(bulletin_id, [])
        if not article_ids:
            continue

        scored: list[tuple[str, dict, float]] = []
        for article_id in article_ids:
            data = corpus_articles.get(article_id)
            if data is None:
                continue
            titre_lower = data["titre"].lower()
            texte_lower = data["texte"].lower()
            score = 0.0
            for kw in keywords:
                kw_l = kw.lower()
                score += titre_lower.count(kw_l) * 3
                score += texte_lower.count(kw_l)
            for kw in title_keywords:
                kw_l = kw.lower()
                score += titre_lower.count(kw_l) * 3
            scored.append((article_id, data, score))

        # Apply rubrique filter if set
        if rubrique_filter:
            rubrique_match = [
                item for item in scored
                if item[1]["rubrique"].lower() == rubrique_filter
            ]
            selected = rubrique_match if rubrique_match else scored
        else:
            selected = scored

        # Keep scored articles; fall back to all if none score > 0
        positive = [item for item in selected if item[2] > 0]
        results.extend(positive if positive else selected)

    return results


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


def afficher_resultats(
    article_list: list[tuple[str, dict, float]],
    keywords: list,
    mode: str,
) -> None:
    """
    Display article-level search results to stdout.

    Parameters:
    - article_list: list of (article_id, article_data, score) from filtrer_articles_pertinents
    - keywords: list of keywords for snippet generation
    - mode: "classe" (sort by score desc) or "booleen" (sort by date desc)
    """
    print(f"\n--- {len(article_list)} résultat(s) trouvé(s) ---\n")

    if not article_list:
        print("Aucun document trouvé.")
        return

    if mode == "classe":
        docs = sorted(article_list, key=lambda x: x[2], reverse=True)
    else:
        def sort_key_date(item):
            _, data, _ = item
            date_str = data.get("date", "")
            try:
                ts = datetime.strptime(date_str, "%d/%m/%Y").timestamp()
                return (0, -ts)
            except (ValueError, TypeError):
                return (1, 0)
        docs = sorted(article_list, key=sort_key_date)

    for i, (article_id, data, score) in enumerate(docs, start=1):
        titre = data.get("titre", "")
        date = data.get("date", "")
        rubrique = data.get("rubrique", "")
        texte = data.get("texte", "")

        score_str = f"    score: {score:.1f}" if mode == "classe" else ""
        print(f"[{i}] Article #{article_id} | Date: {date} | Rubrique: {rubrique}{score_str}")
        print(f"    Titre : {titre}")

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
    corpus_articles, bulletin_to_articles = charger_corpus_articles(CORPUS_FILE)

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

        # Evaluate query → bulletin IDs
        doc_ids = evaluer_requete_recursive(requete, tf_idf_dict, anti_list, index_inverse)

        # Extract keywords and rubrique for article-level resolution
        _, global_filters = extraire_filtres_globaux(requete)
        rubrique_filter = global_filters.get("rubrique")

        req_norm, upper_kw = normaliser_texte(requete, key_word_traite=True)
        meta = pipeline_traitement_requete(req_norm, tf_idf_dict, anti_list, upper_kw)
        LOGICAL_OPS = {"et", "ou", "sans", "mais", "pas", "non"}
        keywords = [k for k in meta.get("key_word", []) if k not in LOGICAL_OPS]
        title_keywords = meta.get("title_keywords", [])

        # Resolve bulletins → articles
        article_list = filtrer_articles_pertinents(
            doc_ids, keywords, title_keywords, rubrique_filter,
            corpus_articles, bulletin_to_articles,
        )

        afficher_resultats(article_list, keywords, mode)


if __name__ == "__main__":
    lancer_moteur()
