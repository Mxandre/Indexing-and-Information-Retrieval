"""Moteur de recherche booléen — LO17 TD6.

Ce module :
1. Lit une requête en langage naturel saisie par l'utilisateur ;
2. Utilise ``traitement_requete.py`` (TD5) pour structurer la requête
   (mots-clés corrigés, dates, rubriques, opérateurs logiques) ;
3. Charge les fichiers inverses générés au TD3 ;
4. Traduit la requête structurée en opérations sur l'index inversé ;
5. Renvoie la liste des identifiants des documents répondant aux
   critères selon un modèle booléen (ou booléen classé) ;
6. Fournit une interface terminal interactive.
"""

from __future__ import annotations

import sys
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

# Ajout du répertoire TD5 au path pour pouvoir importer traitement_requete
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir / "TD5"))

from traitement_requete import (  # noqa: E402
    normaliser_texte,
    pipeline_traitement_requete,
    PICKLE_TF_IDF_FILE,
    PICKLE_ANTI_LIST,
    load_lemmatisatioin_file,
    get_rubrique,
    RUBRIQUE_FILE,
)

# Chemins vers les fichiers inverses générés au TD3
INDEX_INVERSE_FULL_FILE = parent_dir / "TD3" / "inverse_index_full.txt"
CORPUS_FILE = parent_dir / "TD2" / "corpus.xml"

MOIS_FR = {
    "janvier": "01", "février": "02", "fevrier": "02", "mars": "03",
    "avril": "04", "mai": "05", "juin": "06", "juillet": "07",
    "août": "08", "aout": "08", "septembre": "09", "octobre": "10",
    "novembre": "11", "décembre": "12", "decembre": "12",
}


# ---------------------------------------------------------------------------
# Chargement des fichiers inverses (TD3) et du corpus (TD2)
# ---------------------------------------------------------------------------

def charger_index(filepath: Path) -> dict:
    """Charge l'index inverse au format texte généré au TD3.

    Format en entrée (généré par ``TD3/3_1.py``) :
        ``lemma<TAB>doc_id.zone: freq, doc_id.zone: freq, ...``

    Renvoie un dictionnaire ``{lemma: {doc_id: {zone: freq}}}`` qui
    conserve la fréquence par zone. Cette structure permet :
      * une recherche booléenne classique (présence ou non) ;
      * une recherche booléenne classée (pondération par fréquence et
        par zone, par exemple titre vs texte).
    """
    index: dict = {}
    if not filepath.exists():
        print(f"[ATTENTION] Index introuvable : {filepath}")
        return index

    # Cas où le fichier est un pickle (compatibilité)
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    with open(filepath, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.rstrip("\n")
            if not ligne or "\t" not in ligne:
                continue
            lemme, postings = ligne.split("\t", 1)
            postings_dict: dict = defaultdict(dict)
            for entry in postings.split(","):
                entry = entry.strip()
                if not entry or ":" not in entry:
                    continue
                cle, freq_str = entry.rsplit(":", 1)
                cle = cle.strip()
                try:
                    freq = int(freq_str.strip())
                except ValueError:
                    continue
                if "." in cle:
                    doc_id, zone = cle.split(".", 1)
                    postings_dict[doc_id][zone] = freq
                else:
                    postings_dict[cle]["_"] = freq
            index[lemme] = dict(postings_dict)
    return index


def charger_corpus(filepath: Path) -> dict:
    """Charge les métadonnées par document depuis le corpus XML."""
    corpus: dict = {}
    if not filepath.exists():
        print(f"[INFO] Corpus introuvable : {filepath} — filtrage limité.")
        return corpus
    try:
        tree = ET.parse(filepath)
    except ET.ParseError as exc:
        print(f"[ATTENTION] Corpus illisible ({exc}).")
        return corpus

    for doc in tree.getroot().findall("document"):
        # L'identifiant peut se trouver dans <article> (corpus brut) ou
        # <bulletin> (corpus filtré). On essaie les deux.
        for nom_id in ("article", "bulletin"):
            id_elem = doc.find(nom_id)
            if id_elem is not None and id_elem.text and id_elem.text.strip():
                doc_id = id_elem.text.strip()
                break
        else:
            continue

        date_text = (doc.findtext("date") or "").strip()
        rubrique = (doc.findtext("rubrique") or "").strip()
        titre = (doc.findtext("titre") or "").strip()
        texte = (doc.findtext("texte") or "").strip()
        a_image = doc.find("images") is not None or doc.find("image") is not None

        date_obj = None
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                date_obj = datetime.strptime(date_text, fmt)
                break
            except ValueError:
                continue

        corpus[doc_id] = {
            "id": doc_id,
            "date": date_text,
            "date_obj": date_obj,
            "rubrique": rubrique,
            "titre": titre,
            "texte": texte,
            "image": a_image,
        }
    return corpus


# ---------------------------------------------------------------------------
# Opérations primitives sur l'index inversé
# ---------------------------------------------------------------------------

def docs_pour_lemme(index: dict, lemme: str, zones: list[str] | None = None) -> set:
    """Renvoie l'ensemble des doc_ids associés à ``lemme``.

    Si ``zones`` est fourni, on ne garde que les documents où le lemme
    apparaît dans l'une de ces zones.
    """
    info = index.get(lemme)
    if not info:
        return set()
    if zones is None:
        return set(info.keys())
    return {d for d, zone_freqs in info.items() if any(z in zone_freqs for z in zones)}


def freq_pour_lemme(index: dict, lemme: str, doc_id: str,
                    zones: list[str] | None = None) -> int:
    """Renvoie la fréquence du lemme dans le document, agrégée sur les zones."""
    info = index.get(lemme, {}).get(doc_id, {})
    if not info:
        return 0
    if zones is None:
        return sum(info.values())
    return sum(info.get(z, 0) for z in zones)


def tous_les_documents(index: dict) -> set:
    """Renvoie l'ensemble de tous les doc_ids présents dans l'index."""
    docs: set = set()
    for postings in index.values():
        docs.update(postings.keys())
    return docs


# ---------------------------------------------------------------------------
# Normalisation et filtres par champ structurel
# ---------------------------------------------------------------------------

def _date_iso(brute: str) -> str | None:
    """Convertit une date brute en chaîne ISO (``YYYY``, ``YYYY-MM`` ou ``YYYY-MM-DD``)."""
    if not brute:
        return None
    brute = brute.lower().strip()
    mois_alt = "|".join(MOIS_FR)

    m = re.fullmatch(rf"(\d{{1,2}})\s+({mois_alt})\s+(\d{{4}})", brute)
    if m:
        return f"{m.group(3)}-{MOIS_FR[m.group(2)]}-{int(m.group(1)):02d}"
    m = re.fullmatch(r"(\d{1,2})[/\-\s]+(\d{1,2})[/\-\s]+(\d{4})", brute)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    m = re.fullmatch(rf"({mois_alt})\s+(\d{{4}})", brute)
    if m:
        return f"{m.group(2)}-{MOIS_FR[m.group(1)]}"
    m = re.fullmatch(r"(\d{1,2})[/\-\s]+(\d{4})", brute)
    if m:
        return f"{m.group(2)}-{int(m.group(1)):02d}"
    m = re.fullmatch(r"(\d{4})", brute)
    if m:
        return m.group(1)
    return None


def filtrer_par_date(corpus: dict, date_info: dict) -> set:
    """Renvoie les doc_ids dont la date satisfait ``date_info``.

    ``date_info`` suit la structure produite par traitement_requete.py :
      * ``{"type": "dmy"|"my"|"y", "value": "..."}``
      * ``{"type": "between_dmy"|"between_my"|"between_y", "start": "...", "end": "..."}``
    """
    if not corpus:
        return set()

    type_d = date_info.get("type", "")
    resultats: set = set()

    if type_d in ("dmy", "my", "y"):
        cible = _date_iso(date_info.get("value", ""))
        if cible is None:
            return set()
        for doc_id, info in corpus.items():
            d = info.get("date_obj")
            if d is None:
                continue
            iso = d.strftime("%Y-%m-%d")
            if iso.startswith(cible):
                resultats.add(doc_id)

    elif type_d in ("between_dmy", "between_my", "between_y"):
        debut = _date_iso(date_info.get("start", ""))
        fin = _date_iso(date_info.get("end", ""))
        if debut is None or fin is None:
            return set()
        if len(debut) == 4:
            debut += "-01-01"
        elif len(debut) == 7:
            debut += "-01"
        if len(fin) == 4:
            fin += "-12-31"
        elif len(fin) == 7:
            fin += "-31"
        for doc_id, info in corpus.items():
            d = info.get("date_obj")
            if d is None:
                continue
            iso = d.strftime("%Y-%m-%d")
            if debut <= iso <= fin:
                resultats.add(doc_id)

    return resultats


def filtrer_par_rubrique(index: dict, corpus: dict, rubriques) -> set:
    """Filtre par rubrique en utilisant d'abord l'index, puis le corpus."""
    if not rubriques:
        return set()
    if isinstance(rubriques, str):
        rubriques = [rubriques]
    cibles = {r.strip().lower() for r in rubriques if r and r.strip()}
    if not cibles:
        return set()

    # 1) lookup direct dans l'index (3_1.py indexe la rubrique en zone exacte)
    res: set = set()
    for cible in cibles:
        res |= docs_pour_lemme(index, cible, zones=["rubrique"])
        res |= docs_pour_lemme(index, cible)  # repli si pas de zone

    # 2) repli via le corpus
    if not res and corpus:
        res = {doc_id for doc_id, info in corpus.items()
               if info.get("rubrique", "").lower() in cibles}
    return res


def filtrer_par_image(index: dict, corpus: dict, presence: bool) -> set:
    """Filtre selon la présence (ou absence) d'images."""
    avec = docs_pour_lemme(index, "has_image")
    if corpus:
        avec_corpus = {d for d, info in corpus.items() if info.get("image")}
        avec = avec | avec_corpus
        univers = set(corpus.keys()) | tous_les_documents(index)
    else:
        univers = tous_les_documents(index)
    return avec if presence else (univers - avec)


def filtrer_par_titre(index: dict, mots_titre: list[str]) -> set:
    """Filtre les documents dont le titre contient tous les mots fournis."""
    if not mots_titre:
        return set()
    res: set | None = None
    for mot in mots_titre:
        docs = docs_pour_lemme(index, mot.lower(), zones=["titre"])
        res = docs if res is None else (res & docs)
    return res or set()


# ---------------------------------------------------------------------------
# Évaluation booléenne (et booléenne classée)
# ---------------------------------------------------------------------------

def evaluer_metadonnees(meta: dict, index: dict, corpus: dict | None = None) -> set:
    """Combine en intersection (ET) les contraintes d'un nœud feuille.

    Les contraintes prises en compte : ``key_word``, ``rubrique``, ``date``,
    ``image``, ``title_keywords``.
    """
    corpus = corpus or {}
    contraintes: list[set] = []

    # 1. Mots-clés (présence dans n'importe quelle zone du document)
    mots = meta.get("key_word", []) or []
    if mots:
        res: set | None = None
        for mot in mots:
            docs = docs_pour_lemme(index, mot)
            res = docs if res is None else (res & docs)
        contraintes.append(res or set())

    # 2. Rubrique
    rub = meta.get("rubrique")
    if rub:
        contraintes.append(filtrer_par_rubrique(index, corpus, rub))

    # 3. Date
    date_info = meta.get("date")
    if isinstance(date_info, dict) and date_info:
        contraintes.append(filtrer_par_date(corpus, date_info))

    # 4. Image
    if "image" in meta and isinstance(meta["image"], bool):
        contraintes.append(filtrer_par_image(index, corpus, meta["image"]))

    # 5. Mots du titre
    titres = meta.get("title_keywords") or []
    if titres:
        contraintes.append(filtrer_par_titre(index, titres))

    if not contraintes:
        return set()

    resultat = contraintes[0]
    for c in contraintes[1:]:
        resultat = resultat & c
    return resultat


def _structurer(requete: str, tf_idf_dict, anti_list, rubriques_connues: list[str]) -> dict:
    """Appelle le pipeline du TD5 et complète la détection de rubrique."""
    req_norm, upper_kw = normaliser_texte(requete, key_word_traite=True)
    meta = defaultdict(list)
    # Détection des rubriques connues présentes dans la requête
    for rub in rubriques_connues:
        if rub and rub.lower() in req_norm:
            if rub not in meta["rubrique"]:
                meta["rubrique"].append(rub)
    meta = pipeline_traitement_requete(req_norm, meta, tf_idf_dict, anti_list, upper_kw)
    return meta


def evaluer_requete_recursive(requete: str, tf_idf_dict, anti_list,
                              index: dict, corpus: dict | None = None,
                              rubriques_connues: list[str] | None = None) -> set:
    """Évalue récursivement une requête en respectant les opérateurs logiques.

    Opérateurs reconnus (issus du TD5) : ``et``, ``ou``, ``sans`` (et
    variantes négatives ``mais pas``, ``non pas``, ``et non pas``).
    """
    rubriques_connues = rubriques_connues or []
    meta = _structurer(requete, tf_idf_dict, anti_list, rubriques_connues)

    if "operateur" in meta and "part1" in meta and "part2" in meta:
        op = str(meta["operateur"]).lower().strip()
        a = evaluer_requete_recursive(meta["part1"], tf_idf_dict, anti_list,
                                      index, corpus, rubriques_connues)
        b = evaluer_requete_recursive(meta["part2"], tf_idf_dict, anti_list,
                                      index, corpus, rubriques_connues)
        if op == "et":
            return a & b
        if op == "ou":
            return a | b
        if op in {"sans", "mais pas", "non pas", "et non pas"}:
            return a - b

    return evaluer_metadonnees(meta, index, corpus)


def calculer_scores(meta: dict, index: dict, doc_ids: set) -> dict:
    """Score booléen classé : pondération zone-titre × 2 + zone-texte."""
    scores: dict[str, float] = defaultdict(float)
    mots = meta.get("key_word", []) or []
    titres = meta.get("title_keywords") or []
    for doc_id in doc_ids:
        s = 0.0
        for mot in mots:
            s += freq_pour_lemme(index, mot, doc_id, zones=["titre"]) * 2.0
            s += freq_pour_lemme(index, mot, doc_id, zones=["texte"])
        for mot in titres:
            s += freq_pour_lemme(index, mot.lower(), doc_id, zones=["titre"]) * 3.0
        scores[doc_id] = s
    return scores


# ---------------------------------------------------------------------------
# Interface utilisateur terminal interactive
# ---------------------------------------------------------------------------

AIDE = """
Commandes :
  :aide, :help        — afficher cette aide
  :debug              — basculer l'affichage de la requête structurée
  :rang               — basculer le tri par score (booléen classé)
  :limite N           — fixer la limite d'affichage (défaut 20)
  :detail ID          — afficher le détail du document ID
  :rubriques          — lister les rubriques connues
  :quitter            — quitter le moteur

Exemples de requêtes :
  Articles sur l'intelligence artificielle entre 2010 et 2015
  Articles de la rubrique recherche dont le titre contient innovation
  Articles parlant de robots avec des images
  Articles sur la santé sans publicité
"""


def _afficher_resultat(corpus: dict, doc_id: str, score: float | None = None) -> None:
    info = corpus.get(doc_id, {})
    titre = info.get("titre") or "(sans titre)"
    rub = info.get("rubrique") or "?"
    date = info.get("date") or "?"
    if len(titre) > 90:
        titre = titre[:87] + "..."
    score_txt = f"  score={score:.2f}" if score is not None else ""
    print(f"  • {doc_id:>5}{score_txt}  [{rub} - {date}]  {titre}")


def _commande_detail(corpus: dict, doc_id: str) -> None:
    info = corpus.get(doc_id)
    if not info:
        print(f"  Document {doc_id} introuvable.")
        return
    print()
    print(f"--- Document {doc_id} ---")
    print(f"Date     : {info.get('date', '?')}")
    print(f"Rubrique : {info.get('rubrique', '?')}")
    print(f"Titre    : {info.get('titre', '?')}")
    texte = info.get("texte", "") or ""
    apercu = texte[:500] + ("..." if len(texte) > 500 else "")
    print(f"Texte    : {apercu}")
    print()


def lancer_moteur() -> None:
    print("=" * 64)
    print("        MOTEUR DE RECHERCHE BOOLÉEN — LO17 TD6")
    print("=" * 64)

    # 1. Modèles lexicaux (TF-IDF & anti-dictionnaire) — TD5
    if not PICKLE_TF_IDF_FILE.exists() or not PICKLE_ANTI_LIST.exists():
        print("Génération des fichiers TF-IDF et anti-dictionnaire...")
        load_lemmatisatioin_file()

    with open(PICKLE_TF_IDF_FILE, "rb") as f:
        tf_idf_dict = pickle.load(f)
    with open(PICKLE_ANTI_LIST, "rb") as f:
        anti_list = pickle.load(f)

    # 2. Index inverse (TD3)
    print("Chargement de l'index inverse...")
    index = charger_index(INDEX_INVERSE_FULL_FILE)
    print(f"  → {len(index)} lemmes indexés")

    # 3. Corpus pour les filtres date/rubrique/image
    print("Chargement du corpus...")
    corpus = charger_corpus(CORPUS_FILE)
    print(f"  → {len(corpus)} documents")

    # 4. Liste des rubriques connues (utilisée pour la détection)
    rubriques_connues: list[str] = []
    if RUBRIQUE_FILE.exists():
        try:
            rubriques_connues = get_rubrique(RUBRIQUE_FILE)
        except Exception:
            rubriques_connues = []
    if not rubriques_connues and corpus:
        rubriques_connues = sorted({info.get("rubrique", "") for info in corpus.values() if info.get("rubrique")})
    print(f"  → {len(rubriques_connues)} rubriques connues")

    print()
    print("Tapez ':aide' pour la liste des commandes, ':quitter' pour sortir.")
    print()

    debug = False
    rang = False
    limite = 20

    while True:
        try:
            entree = input("requête> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not entree:
            continue

        bas = entree.lower()
        if bas in {":quitter", ":quit", ":q", "quitter", "exit"}:
            break
        if bas in {":aide", ":help", "?"}:
            print(AIDE)
            continue
        if bas == ":debug":
            debug = not debug
            print(f"  debug = {debug}")
            continue
        if bas == ":rang":
            rang = not rang
            print(f"  rang   = {rang} ({'booléen classé' if rang else 'booléen strict'})")
            continue
        if bas.startswith(":limite"):
            try:
                limite = max(1, int(entree.split()[1]))
                print(f"  limite = {limite}")
            except (IndexError, ValueError):
                print("  Usage : :limite N")
            continue
        if bas.startswith(":detail"):
            try:
                doc_id = entree.split()[1].strip()
            except IndexError:
                print("  Usage : :detail ID")
                continue
            _commande_detail(corpus, doc_id)
            continue
        if bas == ":rubriques":
            if not rubriques_connues:
                print("  Aucune rubrique connue.")
            else:
                print("  " + ", ".join(rubriques_connues))
            continue

        # Évaluation de la requête
        try:
            if debug:
                meta = _structurer(entree, tf_idf_dict, anti_list, rubriques_connues)
                print("  [requête structurée]", dict(meta))

            doc_ids = evaluer_requete_recursive(
                entree, tf_idf_dict, anti_list, index, corpus, rubriques_connues
            )
        except Exception as exc:
            print(f"  Erreur lors de l'évaluation : {exc}")
            continue

        if not doc_ids:
            print("  Aucun document trouvé.")
            print()
            continue

        print(f"  → {len(doc_ids)} document(s) trouvé(s).")
        if rang:
            meta = _structurer(entree, tf_idf_dict, anti_list, rubriques_connues)
            scores = calculer_scores(meta, index, doc_ids)
            tries = sorted(doc_ids, key=lambda d: (-scores.get(d, 0.0), d))
            for doc_id in tries[:limite]:
                _afficher_resultat(corpus, doc_id, scores.get(doc_id, 0.0))
        else:
            for doc_id in sorted(doc_ids)[:limite]:
                _afficher_resultat(corpus, doc_id)

        if len(doc_ids) > limite:
            reste = len(doc_ids) - limite
            print(f"  ... ({reste} de plus — utilisez ':limite N' pour ajuster)")
        print()

    print("Arrêt du moteur. Au revoir.")


if __name__ == "__main__":
    lancer_moteur()
