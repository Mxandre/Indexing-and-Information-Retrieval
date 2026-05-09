"""Moteur de recherche booléen/classé pour le corpus LO17.

Charge l'index inverse et le corpus XML filtré, évalue des requêtes en langage
naturel via le pipeline TD5, résout les bulletins en articles individuels et
affiche les résultats avec extraits contextuels.

Le modèle de requête repose sur une DNF labellisée (``key_word_groups``) :
chaque groupe est un dict ``{"title": [...], "content": [...]}`` représentant
une conjonction de contraintes. Les groupes sont combinés en disjonction (OU).
Les contraintes ``title`` exigent la présence du terme dans la zone `titre` de
l'index ; les contraintes ``content`` acceptent titre ou texte.
"""

import pickle
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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
    """Charge l'index inverse depuis un fichier texte.

    Chaque ligne a le format ``lemme<TAB>entrée, entrée, …`` où chaque entrée
    est ``doc_id.zone: fréquence`` (ex. ``273.titre: 1, 273.texte: 3``).

    Args:
        filepath (Path): Chemin du fichier d'index.

    Returns:
        dict: Index de structure ``{lemme: {doc_id: {zone: fréquence}}}``.
        Retourne ``{}`` si le fichier est absent.
    """
    if not filepath.exists():
        print(f"Avertissement : fichier d'index introuvable — {filepath}")
        return {}

    index = {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
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
                        # Format de l'entrée : doc_id.zone: fréquence
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
                    continue

    except Exception as e:
        print(f"Erreur lors du chargement de l'index depuis {filepath} : {e}")
        return {}

    return index


MOIS_FR = {
    "janvier": 1, "février": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
}


def _parse_dmy(s: str) -> datetime | None:
    """Analyse une chaîne de date au format numérique ou texte français.

    Tente d'abord le format ``%d/%m/%Y``, puis le format texte
    ``j mois_littéral année`` (ex. ``3 mars 2013``).

    Args:
        s (str): Chaîne de date à analyser.

    Returns:
        datetime | None: Objet datetime correspondant, ou ``None`` si non reconnu.
    """
    try:
        return datetime.strptime(s, "%d/%m/%Y")
    except ValueError:
        pass
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", s.strip(), re.IGNORECASE)
    if m:
        month = MOIS_FR.get(m.group(2).lower())
        if month:
            try:
                return datetime(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                pass
    return None


def evaluer_metadonnees(metadonnees: dict, index_inverse: dict) -> dict[str, float]:
    """Score les documents selon les contraintes de la requête structurée.

    Applique la DNF labellisée (``key_word_groups``) puis filtre par rubrique
    et/ou date. Le score d'un document est la somme des contributions par
    groupe (OU) ; au sein d'un groupe, tous les termes doivent être présents
    (ET). Les termes ``title`` n'acceptent que la zone ``titre`` (x3) ;
    les termes ``content`` acceptent ``titre`` (x3) ou ``texte`` (x1).

    Args:
        metadonnees (dict): Résultat de ``pipeline_traitement_requete``.
        index_inverse (dict): Index inverse chargé.

    Returns:
        dict[str, float]: Scores ``{doc_id: score}``. Retourne ``{}`` si tous
        les filtres sont absents.
    """
    keywords = metadonnees.get("key_word", [])
    title_keywords = metadonnees.get("title_keywords", [])
    rubrique = metadonnees.get("rubrique", None)
    date_info = metadonnees.get("date", None)

    if not keywords and not title_keywords and rubrique is None and date_info is None:
        return {}

    # Construire les groupes DNF ; compatibilité ascendante avec les champs plats
    key_word_groups = metadonnees.get("key_word_groups")
    if not key_word_groups and (keywords or title_keywords):
        key_word_groups = [{"title": list(title_keywords), "content": list(keywords)}]

    # ------------------------------------------------------------------ #
    # Étape 1 : score par mots-clés — DNF labellisée                      #
    #   - contrainte title  → zone titre uniquement (score titre × 3)      #
    #   - contrainte content → titre OU texte (score titre×3 + texte×1)   #
    #   - ET au sein d'un groupe, OU (somme) entre groupes                 #
    # ------------------------------------------------------------------ #
    keyword_scores: dict[str, float] | None = None

    if key_word_groups:
        keyword_scores = {}
        for group in key_word_groups:
            g_scores: dict[str, float] | None = None

            for kw in group.get("title", []):
                kw_docs = index_inverse.get(kw, {})
                kw_score: dict[str, float] = {
                    doc_id: zones.get("titre", 0) * 3
                    for doc_id, zones in kw_docs.items()
                    if zones.get("titre", 0) > 0
                }
                if g_scores is None:
                    g_scores = kw_score
                else:
                    g_scores = {
                        doc_id: s + kw_score[doc_id]
                        for doc_id, s in g_scores.items()
                        if doc_id in kw_score
                    }

            for kw in group.get("content", []):
                kw_docs = index_inverse.get(kw, {})
                kw_score = {
                    doc_id: zones.get("titre", 0) * 3 + zones.get("texte", 0)
                    for doc_id, zones in kw_docs.items()
                }
                if g_scores is None:
                    g_scores = kw_score
                else:
                    g_scores = {
                        doc_id: s + kw_score[doc_id]
                        for doc_id, s in g_scores.items()
                        if doc_id in kw_score
                    }

            if g_scores:
                for doc_id, s in g_scores.items():
                    keyword_scores[doc_id] = keyword_scores.get(doc_id, 0.0) + s

    
    # Étape 2 : filtre rubrique                                            
    rubrique_docs: set[str] | None = None
    if rubrique is not None:
        rubrique_lower = rubrique.lower()
        rub_entries = {
            doc_id: zones
            for doc_id, zones in index_inverse.get(rubrique_lower, {}).items()
            if "rubrique" in zones
        }
        rubrique_docs = set(rub_entries.keys())

    
    # Étape 3 : filtre date                                                
    
    date_docs: set[str] | None = None
    if date_info is not None:
        dtype = date_info.get("type", "")

        if dtype == "dmy":
            val = date_info.get("value", "")
            date_docs = set(index_inverse.get(val, {}).keys())
            if not date_docs:
                dt = _parse_dmy(val)
                if dt:
                    date_docs = set(index_inverse.get(dt.strftime("%d/%m/%Y"), {}).keys())

        elif dtype == "y":
            year_val = date_info["value"]
            date_docs = set()
            for key, docs in index_inverse.items():
                try:
                    dt = datetime.strptime(key, "%d/%m/%Y")
                    if str(dt.year) == year_val:
                        date_docs.update(docs.keys())
                except ValueError:
                    pass

        elif dtype == "my":
            my_str = date_info["value"]
            date_docs = set()
            target: datetime | None = None
            for fmt in ("%m/%Y", "%m-%Y", "%m %Y"):
                try:
                    target = datetime.strptime(my_str, fmt)
                    break
                except ValueError:
                    continue
            if target is None:
                # Format texte : « septembre 2012 »
                m2 = re.match(r"(\w+)\s+(\d{4})", my_str.strip(), re.IGNORECASE)
                if m2:
                    month = MOIS_FR.get(m2.group(1).lower())
                    if month:
                        try:
                            target = datetime(int(m2.group(2)), month, 1)
                        except ValueError:
                            pass
            if target is not None:
                for key, docs in index_inverse.items():
                    try:
                        dt = datetime.strptime(key, "%d/%m/%Y")
                        if dt.month == target.month and dt.year == target.year:
                            date_docs.update(docs.keys())
                    except ValueError:
                        pass

        elif dtype == "between_dmy":
            start_dt = _parse_dmy(date_info.get("start", ""))
            end_dt = _parse_dmy(date_info.get("end", ""))

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
    
    # Étape 4 : combinaison des résultats                                 
    
    if keyword_scores is not None:
        result = keyword_scores
        if rubrique_docs is not None:
            result = {doc_id: score for doc_id, score in result.items() if doc_id in rubrique_docs}
        if date_docs is not None:
            result = {doc_id: score for doc_id, score in result.items() if doc_id in date_docs}
        return result
    else:
        # Pas de mots-clés : combiner rubrique et/ou date avec un score nul
        combined: set[str] | None = None
        if rubrique_docs is not None:
            combined = rubrique_docs
        if date_docs is not None:
            combined = date_docs if combined is None else combined & date_docs
        if combined is None:
            return {}
        return {doc_id: 0.0 for doc_id in combined}


def _soustraire_exclus(docs: dict, key_word_exclu: list, index_inverse: dict) -> dict:
    """Retire des résultats les documents contenant un mot exclu.

    Args:
        docs (dict[str, float]): Scores courants ``{doc_id: score}``.
        key_word_exclu (list[str]): Mots dont la présence exclut un document.
        index_inverse (dict): Index inverse chargé.

    Returns:
        dict[str, float]: Scores filtrés.
    """
    if not key_word_exclu:
        return docs
    exclu_docs: set = set()
    for kw in key_word_exclu:
        exclu_docs.update(index_inverse.get(kw, {}).keys())
    return {doc_id: score for doc_id, score in docs.items() if doc_id not in exclu_docs}


def evaluer_requete_complet(
    requete_texte: str,
    tf_idf_dict: dict,
    anti_list: list,
    index_inverse: dict,
) -> tuple[set, dict]:
    """Exécute le pipeline TD5 puis évalue la requête structurée contre l'index.

    Les contraintes de zone titre sont appliquées par groupe dans
    ``evaluer_metadonnees`` (aucun post-filtrage séparé n'est nécessaire).

    Args:
        requete_texte (str): Requête brute en langage naturel.
        tf_idf_dict (dict): Scores TF-IDF.
        anti_list (list): Anti-dictionnaire.
        index_inverse (dict): Index inverse chargé.

    Returns:
        tuple[set[str], dict]: Paire ``(doc_ids, meta)`` permettant au
        code appelant de réutiliser ``meta`` sans relancer le pipeline.
    """
    req_norm, upper_kw = normaliser_texte(requete_texte, key_word_traite=True)
    meta = pipeline_traitement_requete(req_norm, anti_list, upper_kw)
    docs = evaluer_metadonnees(meta, index_inverse)
    docs = _soustraire_exclus(docs, meta.get("key_word_exclu", []), index_inverse)
    return set(docs.keys()), meta


def evaluer_requete_recursive(
    requete_texte: str,
    tf_idf_dict: dict,
    anti_list: list,
    index_inverse: dict,
) -> set:
    """Enveloppe rétrocompatible retournant uniquement l'ensemble des doc_id.

    Args:
        requete_texte (str): Requête brute en langage naturel.
        tf_idf_dict (dict): Scores TF-IDF.
        anti_list (list): Anti-dictionnaire.
        index_inverse (dict): Index inverse chargé.

    Returns:
        set[str]: Identifiants de bulletins retrouvés.
    """
    doc_ids, _ = evaluer_requete_complet(requete_texte, tf_idf_dict, anti_list, index_inverse)
    return doc_ids


def charger_corpus(filepath: Path) -> dict:
    """Charge le corpus depuis un fichier XML (groupé par bulletin).

    Structure XML attendue : ``<corpus><document><bulletin>…</bulletin>…</document></corpus>``.
    Si plusieurs documents partagent le même bulletin, leurs rubriques et textes
    sont fusionnés.

    Args:
        filepath (Path): Chemin du fichier XML.

    Returns:
        dict: Index ``{bulletin_id: {"titre", "date", "rubriques", "texte"}}``.
        Retourne ``{}`` si le fichier est absent.
    """
    if not filepath.exists():
        print(f"Avertissement : fichier corpus introuvable — {filepath}")
        return {}

    corpus = {}

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for document in root.findall("document"):
            bulletin_elem = document.find("bulletin")
            if bulletin_elem is None or not bulletin_elem.text:
                continue
            bulletin_id = bulletin_elem.text.strip()
            if not bulletin_id:
                continue

            titre = ""
            titre_elem = document.find("titre")
            if titre_elem is not None and titre_elem.text:
                titre = titre_elem.text.strip()

            date = ""
            date_elem = document.find("date")
            if date_elem is not None and date_elem.text:
                date = date_elem.text.strip()

            rubrique = ""
            rubrique_elem = document.find("rubrique")
            if rubrique_elem is not None and rubrique_elem.text:
                rubrique = rubrique_elem.text.strip()

            texte = ""
            texte_elem = document.find("texte")
            if texte_elem is not None and texte_elem.text:
                texte = texte_elem.text.strip()

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
        print(f"Erreur lors du chargement du corpus depuis {filepath} : {e}")
        return {}

    return corpus


def charger_corpus_articles(filepath: Path) -> tuple[dict, dict]:
    """Charge le corpus et construit l'index article → bulletin.

    Args:
        filepath (Path): Chemin du fichier XML.

    Returns:
        tuple[dict, dict]: Paire ``(corpus_articles, bulletin_to_articles)`` où
        ``corpus_articles`` est ``{article_id: {article, bulletin, date, rubrique,
        titre, texte}}`` et ``bulletin_to_articles`` est ``{bulletin_id: [article_id, ...]}``.
        Retourne ``({}, {})`` si le fichier est absent.
    """
    if not filepath.exists():
        print(f"Avertissement : fichier corpus introuvable — {filepath}")
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
            bulletin_id = (
                bulletin_elem.text.strip()
                if bulletin_elem is not None and bulletin_elem.text
                else ""
            )

            titre_elem = document.find("titre")
            titre = titre_elem.text.strip() if titre_elem is not None and titre_elem.text else ""

            date_elem = document.find("date")
            date = date_elem.text.strip() if date_elem is not None and date_elem.text else ""

            rubrique_elem = document.find("rubrique")
            rubrique = (
                rubrique_elem.text.strip()
                if rubrique_elem is not None and rubrique_elem.text
                else ""
            )

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
        print(f"Erreur lors du chargement des articles depuis {filepath} : {e}")
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
    """Résout les bulletins en articles individuels et les classe par pertinence.

    Pour chaque bulletin : score les articles par présence des mots-clés dans
    le titre (×3) et le texte (×1). Si un filtre de rubrique est actif, préfère
    les articles correspondants (repli sur tous si aucun ne correspond). Ne
    retient que les articles avec un score > 0 ; repli sur tous si aucun.

    Args:
        bulletin_ids (set[str]): Bulletins retrouvés par ``evaluer_requete_complet``.
        keywords (list[str]): Mots-clés de contenu pour le scoring.
        title_keywords (list[str]): Mots-clés de titre pour le scoring.
        rubrique_filter (str | None): Rubrique exacte à privilégier.
        corpus_articles (dict): Articles indexés par ID.
        bulletin_to_articles (dict): Correspondance bulletin → liste d'articles.

    Returns:
        list[tuple[str, dict, float]]: Liste de triplets
        ``(article_id, données_article, score)``.
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

        if rubrique_filter:
            rubrique_match = [
                item for item in scored
                if item[1]["rubrique"].lower() == rubrique_filter
            ]
            selected = rubrique_match if rubrique_match else scored
        else:
            selected = scored

        positive = [item for item in selected if item[2] > 0]
        results.extend(positive if positive else selected)

    return results


def generer_snippets(texte: str, keywords: list, window: int = 40) -> list:
    """Trouve toutes les occurrences des mots-clés et retourne des extraits contextuels.

    Les fenêtres se chevauchant sont fusionnées. Les mots-clés sont mis en
    évidence entre crochets. En l'absence d'occurrence, retourne les 120
    premiers caractères.

    Args:
        texte (str): Texte complet de l'article.
        keywords (list[str]): Mots-clés à repérer.
        window (int): Nombre de caractères de contexte autour de chaque occurrence.

    Returns:
        list[str]: Extraits textuels avec mots-clés entre crochets.
    """
    if not keywords or not texte:
        return [texte[:120] + "..."] if texte else []

    texte_lower = texte.lower()

    # Collecte de toutes les occurrences avec leur fenêtre
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

    occurrences.sort(key=lambda x: x[0])

    # Fusion des fenêtres qui se chevauchent
    merged = []
    cur_start, cur_end = occurrences[0]
    for start, end in occurrences[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    # Construction des extraits avec mise en évidence entre crochets
    snippets = []
    for win_start, win_end in merged:
        chunk = texte[win_start:win_end]

        # Collecte de tous les spans de correspondance avant de modifier le chunk
        # (pour éviter l'imbrication de crochets)
        all_matches = []
        for kw in set(keywords):
            for m in re.finditer(re.escape(kw), chunk, re.IGNORECASE):
                all_matches.append((m.start(), m.end(), m.group(0)))
        all_matches.sort(key=lambda x: x[0])

        result_parts = []
        pos = 0
        for start, end, found in all_matches:
            if start < pos:
                continue  # Ignorer les correspondances qui se chevauchent
            result_parts.append(chunk[pos:start])
            result_parts.append(f"[{found}]")
            pos = end
        result_parts.append(chunk[pos:])
        chunk = "".join(result_parts)

        prefix = "..." if win_start > 0 else ""
        suffix = "..." if win_end < len(texte) else ""
        snippets.append(prefix + chunk + suffix)

    return snippets


def afficher_resultats(
    article_list: list[tuple[str, dict, float]],
    keywords: list,
    mode: str,
) -> None:
    """Affiche les résultats de recherche sur la sortie standard.

    Args:
        article_list (list[tuple[str, dict, float]]): Triplets
            ``(article_id, données_article, score)`` issus de
            ``filtrer_articles_pertinents``.
        keywords (list[str]): Mots-clés pour la génération des extraits.
        mode (str): ``"classe"`` (tri par score décroissant) ou ``"booleen"``
            (tri par date décroissante).
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

        score_str = f"    score : {score:.1f}" if mode == "classe" else ""
        print(f"[{i}] Article #{article_id} | Date : {date} | Rubrique : {rubrique}{score_str}")
        print(f"    Titre : {titre}")

        snips = generer_snippets(texte, keywords)
        if snips:
            print("    Extraits :")
            for snip in snips:
                print(f"      {snip}")

        if i < len(docs):
            print()


def lancer_moteur() -> None:
    """Lance la boucle interactive du moteur de recherche.

    Commandes disponibles :
    - ``/mode booleen`` : résultats triés par date décroissante.
    - ``/mode classe``  : résultats triés par score de pertinence.
    - ``/quitter``      : quitter le moteur.
    """
    mode = "classe"
    print("================================================")
    print(f"    MOTEUR DE RECHERCHE — Mode : {mode}")
    print("================================================")
    print("Commandes : /mode booleen | /mode classe | /quitter")

    if not PICKLE_TF_IDF_FILE.exists() or not PICKLE_ANTI_LIST.exists():
        load_lemmatisatioin_file()

    with open(str(PICKLE_TF_IDF_FILE), "rb") as f:
        tf_idf_dict = pickle.load(f)
    with open(str(PICKLE_ANTI_LIST), "rb") as f:
        anti_list = pickle.load(f)

    index_inverse = charger_index(INDEX_INVERSE_FULL_FILE)
    corpus_articles, bulletin_to_articles = charger_corpus_articles(CORPUS_FILE)

    print("Moteur prêt !\n")

    LOGICAL_OPS = {"et", "ou", "sans", "mais", "pas", "non"}

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

        doc_ids, meta = evaluer_requete_complet(requete, tf_idf_dict, anti_list, index_inverse)

        keywords = [k for k in meta.get("key_word", []) if k not in LOGICAL_OPS]
        title_keywords = meta.get("title_keywords", [])
        rubrique_filter = meta.get("rubrique")

        article_list = filtrer_articles_pertinents(
            doc_ids, keywords, title_keywords, rubrique_filter,
            corpus_articles, bulletin_to_articles,
        )

        afficher_resultats(article_list, keywords, mode)


if __name__ == "__main__":
    lancer_moteur()
