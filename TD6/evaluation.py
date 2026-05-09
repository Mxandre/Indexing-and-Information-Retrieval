"""Évaluation qualité et performance du moteur de recherche LO17.

Exécute 10 requêtes de référence (vérité terrain annotée manuellement),
calcule la précision, le rappel et le F1 par requête, mesure le temps de
réponse moyen sur 10 exécutions et génère des graphiques PNG résumant les
résultats.
"""

import pickle
import sys
import time
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir / "TD5"))

from traitement_requete import (
    normaliser_texte,
    pipeline_traitement_requete,
    PICKLE_TF_IDF_FILE,
    PICKLE_ANTI_LIST,
    load_lemmatisatioin_file,
)
from moteur import (
    charger_index,
    charger_corpus_articles,
    evaluer_requete_recursive,
    filtrer_articles_pertinents,
    INDEX_INVERSE_FULL_FILE,
    CORPUS_FILE,
)

# ---------------------------------------------------------------------------
# Vérité terrain : 10 requêtes couvrant des types de contraintes variés.
# Les ensembles ``relevant_articles`` ont été établis par inspection manuelle
# de corpus_filtre.xml. Utiliser ``chercher_candidats()`` pour les compléter.
# ---------------------------------------------------------------------------
GROUND_TRUTH: list[dict] = [
    {
        "id": "Q01",
        # rubrique Horizons Enseignement + « embarqué » dans le texte
        "query": "Afficher la liste des articles qui parlent des systèmes embarqués dans la rubrique Horizons Enseignement",
        "relevant_articles": {"74751"},
    },
    {
        "id": "Q02",
        # tous les articles parus du 03/03/2013 au 04/05/2013 inclus
        "query": "Quels sont les articles parus entre le 3 mars 2013 et le 4 mai 2013",
        "relevant_articles": {
            "72629", "72630", "72631", "72632", "72633", "72634", "72635",
            "72636", "72637", "72932", "72933", "72934", "72935", "72936",
            "72937", "72938", "72939", "72940",
        },
    },
    {
        "id": "Q03",
        # rubrique Focus + « innovation » dans le titre ou le texte
        "query": "Je veux les articles de la rubrique Focus parlant d'innovation",
        "relevant_articles": {
            "67068", "67383", "68273", "68276", "68383", "69533", "69535",
            "70162", "71359", "72392", "72393", "72630", "72933", "73876",
            "74167", "74744", "76507",
        },
    },
    {
        "id": "Q04",
        # rubrique Focus + année 2014 + « santé » dans le titre ou le texte
        "query": "Je veux les articles de 2014 et de la rubrique Focus et parlant de santé",
        "relevant_articles": {"75459", "76507"},
    },
    {
        "id": "Q05",
        # « cnrs » ET « chim » présents dans le titre ou le texte
        "query": "Je veux les articles impliquant le CNRS et qui parlent de chimie",
        "relevant_articles": {
            "67068", "67071", "67558", "67800", "68278", "68280", "68388",
            "68390", "69183", "69184", "70745", "70922", "72632", "72634",
            "72940", "73189", "73436", "74173", "74750", "75066", "75067", "75070",
        },
    },
    {
        "id": "Q06",
        # « airbus » OU « taxibot » présents dans le titre ou le texte
        "query": "Je voudrais les articles qui parlent d'airbus ou du projet Taxibot",
        "relevant_articles": {"67797", "70920", "71617", "72636", "72933", "74745"},
    },
    {
        "id": "Q07",
        # « russie » OU « japon » présents dans le titre ou le texte
        "query": "Quels sont les articles parlant de la Russie ou du Japon",
        "relevant_articles": {
            "67383", "67939", "67942", "67943", "68388", "68642", "69185",
            "70915", "72117", "72396", "73880", "74168", "74746", "75064",
        },
    },
    {
        "id": "Q08",
        # « chim » présent dans le titre (chimie, chimique, chimiste…)
        "query": "Je voudrais les articles dont le titre contient le mot chimie",
        "relevant_articles": {"67392", "67561", "68278", "68390", "74752", "75461"},
    },
    {
        "id": "Q09",
        # année 2011 + « enseign » dans le titre ou le texte
        "query": "Je voudrais les articles de 2011 sur l'enseignement",
        "relevant_articles": {
            "67068", "67071", "67795", "67944", "68277", "68281", "68392", "68393",
        },
    },
    {
        "id": "Q10",
        # « biocarburant » dans le titre OU « bioénerg » dans le texte
        "query": "Quels sont les articles dont le titre contient biocarburant ou le contenu parle des bioénergies",
        "relevant_articles": {"68385", "72121"},
    },
]

LOGICAL_OPS = {"et", "ou", "sans", "mais", "pas", "non"}


def load_resources() -> dict:
    """Charge toutes les ressources du moteur en une seule passe.

    Returns:
        dict: Dictionnaire dépaquetable (``**``) dans ``run_query``, contenant
        ``index_inverse``, ``corpus_articles``, ``bulletin_to_articles``,
        ``tf_idf_dict``, ``anti_list``.
    """
    if not PICKLE_TF_IDF_FILE.exists() or not PICKLE_ANTI_LIST.exists():
        load_lemmatisatioin_file()
    with open(str(PICKLE_TF_IDF_FILE), "rb") as f:
        tf_idf_dict = pickle.load(f)
    with open(str(PICKLE_ANTI_LIST), "rb") as f:
        anti_list = pickle.load(f)
    index_inverse = charger_index(INDEX_INVERSE_FULL_FILE)
    corpus_articles, bulletin_to_articles = charger_corpus_articles(CORPUS_FILE)
    return {
        "index_inverse": index_inverse,
        "corpus_articles": corpus_articles,
        "bulletin_to_articles": bulletin_to_articles,
        "tf_idf_dict": tf_idf_dict,
        "anti_list": anti_list,
    }


def run_query(
    query: str,
    index_inverse: dict,
    corpus_articles: dict,
    bulletin_to_articles: dict,
    tf_idf_dict: dict,
    anti_list: list,
) -> set[str]:
    """Exécute le pipeline complet sur une requête et retourne les IDs d'articles.

    Args:
        query (str): Requête en langage naturel.
        index_inverse (dict): Index inverse chargé.
        corpus_articles (dict): Articles du corpus indexés par ID.
        bulletin_to_articles (dict): Correspondance bulletin → liste d'articles.
        tf_idf_dict (dict): Scores TF-IDF.
        anti_list (list): Anti-dictionnaire.

    Returns:
        set[str]: Ensemble des identifiants d'articles retrouvés.
    """
    bulletin_ids = evaluer_requete_recursive(query, tf_idf_dict, anti_list, index_inverse)
    req_norm, upper_kw = normaliser_texte(query, key_word_traite=True)
    meta = pipeline_traitement_requete(req_norm, tf_idf_dict, anti_list, upper_kw)
    keywords = [k for k in meta.get("key_word", []) if k not in LOGICAL_OPS]
    title_keywords = meta.get("title_keywords", [])
    rubrique_filter = meta.get("rubrique")
    article_list = filtrer_articles_pertinents(
        bulletin_ids, keywords, title_keywords, rubrique_filter,
        corpus_articles, bulletin_to_articles,
    )
    return {aid for aid, _, _ in article_list}


def chercher_candidats(
    corpus_articles: dict,
    *,
    keywords: list[str] = (),
    rubrique: str | None = None,
    year: str | None = None,
    title_keyword: str | None = None,
) -> list[tuple[str, str, str]]:
    """Recherche des articles du corpus selon des critères combinés.

    Outil d'aide à l'annotation de la vérité terrain. Tous les critères sont
    combinés en ET.

    Args:
        corpus_articles (dict): Articles indexés par ID (depuis ``charger_corpus_articles``).
        keywords (list[str]): Mots-clés devant tous être présents dans titre ou texte.
        rubrique (str | None): Préfixe de rubrique à filtrer (insensible à la casse).
        year (str | None): Année de publication (4 chiffres).
        title_keyword (str | None): Sous-chaîne devant apparaître dans le titre.

    Returns:
        list[tuple[str, str, str]]: Liste triée de triplets
        ``(article_id, rubrique, titre)``, également affichée sur stdout.
    """
    results = []
    for art in corpus_articles.values():
        aid = art.get("article", "")
        rub = art.get("rubrique", "")
        date = art.get("date", "")
        titre = art.get("titre", "")
        texte = art.get("texte", "")

        if rubrique and not rub.lower().startswith(rubrique.lower()):
            continue
        if year and date[-4:] != year:
            continue
        if title_keyword and title_keyword.lower() not in titre.lower():
            continue
        if keywords:
            combined = (titre + " " + texte).lower()
            if not all(kw.lower() in combined for kw in keywords):
                continue
        results.append((aid, rub, titre))

    results.sort(key=lambda x: x[0])
    print(f"\n--- {len(results)} candidat(s) ---")
    for aid, rub, titre in results:
        print(f"  #{aid:<6}  [{rub}]  {titre[:72]}")
    return results


def compute_metrics(retrieved: set[str], relevant: set[str]) -> dict:
    """Calcule la précision, le rappel et le F1 entre deux ensembles d'IDs.

    Args:
        retrieved (set[str]): Articles retrouvés par le moteur.
        relevant (set[str]): Articles pertinents selon la vérité terrain.

    Returns:
        dict: Dictionnaire ``{"precision": ..., "recall": ..., "f1": ...}``.
    """
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(retrieved & relevant)
    precision = tp / len(retrieved)
    recall = tp / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def measure_performance(
    ground_truth: list[dict],
    resources: dict,
    runs_per_query: int = 10,
) -> dict[str, float]:
    """Mesure le temps de réponse moyen de chaque requête sur plusieurs exécutions.

    Args:
        ground_truth (list[dict]): Liste des entrées de la vérité terrain.
        resources (dict): Ressources chargées par ``load_resources``.
        runs_per_query (int): Nombre d'exécutions par requête (défaut : 10).

    Returns:
        dict[str, float]: Dictionnaire ``{query_id: temps_moyen_ms}``.
    """
    timing: dict[str, float] = {}
    for entry in ground_truth:
        qid = entry["id"]
        query = entry["query"]
        times = []
        for _ in range(runs_per_query):
            t0 = time.perf_counter()
            run_query(query, **resources)
            times.append(time.perf_counter() - t0)
        avg_ms = (sum(times) / len(times)) * 1000
        timing[qid] = avg_ms

    total_runs = len(ground_truth) * runs_per_query
    grand_avg = sum(timing.values()) / len(timing) if timing else 0.0
    print(f"\nTemps moyen global : {grand_avg:.1f} ms/requête ({total_runs} exécutions)")
    return timing


def print_table(results: list[dict]) -> None:
    """Affiche un tableau ASCII récapitulatif des métriques d'évaluation.

    Args:
        results (list[dict]): Liste des résultats par requête (champs :
            ``id``, ``query``, ``precision``, ``recall``, ``f1``, ``time_ms``,
            ``annotated``).
    """
    if not results:
        print("Aucun résultat (vérité terrain non annotée).")
        return
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"{'ID':<4}  {'Requête':<40}  {'P':>6}  {'R':>6}  {'F1':>6}  {'ms':>7}  Ann.")
    print("-" * 78)
    for r in results:
        q_short = r["query"][:38] + ".." if len(r["query"]) > 38 else r["query"]
        annotated = "✓" if r.get("annotated") else "—"
        print(
            f"{r['id']:<4}  {q_short:<40}  "
            f"{r.get('precision', 0.0):>6.3f}  "
            f"{r.get('recall', 0.0):>6.3f}  "
            f"{r.get('f1', 0.0):>6.3f}  "
            f"{r.get('time_ms', 0.0):>7.1f}  {annotated}"
        )
    print(sep)
    annotated_rows = [r for r in results if r.get("annotated")]
    if annotated_rows:
        avg_p = sum(r["precision"] for r in annotated_rows) / len(annotated_rows)
        avg_r = sum(r["recall"] for r in annotated_rows) / len(annotated_rows)
        avg_f1 = sum(r["f1"] for r in annotated_rows) / len(annotated_rows)
        print(f"{'Moy.':<4}  {'(requêtes annotées)':<40}  {avg_p:>6.3f}  {avg_r:>6.3f}  {avg_f1:>6.3f}")


def plot_results(results: list[dict]) -> None:
    """Sauvegarde les graphiques de précision/rappel et de temps de réponse.

    Génère deux fichiers PNG dans le dossier TD6 :
    - ``eval_precision_recall.png`` : précision, rappel et F1 par requête annotée.
    - ``eval_timing.png`` : temps de réponse moyen par requête.

    Args:
        results (list[dict]): Liste des résultats par requête.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib non disponible — graphiques non générés.")
        return

    annotated = [r for r in results if r.get("annotated")]
    ids_all = [r["id"] for r in results]
    timings = [r.get("time_ms", 0.0) for r in results]

    # Graphique 1 : Précision, Rappel, F1 (requêtes annotées uniquement)
    if annotated:
        ids_ann = [r["id"] for r in annotated]
        precisions = [r["precision"] for r in annotated]
        recalls = [r["recall"] for r in annotated]
        f1s = [r["f1"] for r in annotated]
        x = range(len(ids_ann))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(ids_ann) * 1.2), 5))
        ax.bar([i - w for i in x], precisions, w, label="Précision", color="#4C72B0")
        ax.bar(list(x), recalls, w, label="Rappel", color="#DD8452")
        ax.bar([i + w for i in x], f1s, w, label="F1", color="#55A868")
        ax.set_xticks(list(x))
        ax.set_xticklabels(ids_ann)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Précision, Rappel et F1 par requête")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        out1 = current_dir / "eval_precision_recall.png"
        fig.savefig(str(out1), dpi=150)
        plt.close(fig)
        print(f"[INFO] {out1}")

    # Graphique 2 : Temps de réponse (toutes les requêtes)
    fig, ax = plt.subplots(figsize=(max(8, len(ids_all) * 1.1), 4))
    ax.bar(ids_all, timings, color="#4C72B0")
    ax.set_ylabel("Temps moyen (ms)")
    ax.set_title("Temps de réponse moyen par requête (10 exécutions chacune)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out2 = current_dir / "eval_timing.png"
    fig.savefig(str(out2), dpi=150)
    plt.close(fig)
    print(f"[INFO] {out2}")


def main() -> None:
    """Point d'entrée principal : évalue la qualité et la performance du moteur."""
    print("Chargement des ressources...")
    resources = load_resources()

    unannotated = [q["id"] for q in GROUND_TRUTH if not q["relevant_articles"]]
    if unannotated:
        print(f"\n[WARN] Vérité terrain non annotée : {', '.join(unannotated)}")
        print("       Remplissez 'relevant_articles' dans GROUND_TRUTH, puis relancez.")

    # Évaluation de la qualité
    results = []
    for q in GROUND_TRUTH:
        annotated = bool(q["relevant_articles"])
        if annotated:
            retrieved = run_query(q["query"], **resources)
            metrics = compute_metrics(retrieved, q["relevant_articles"])
        else:
            metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        results.append({
            "id": q["id"],
            "query": q["query"],
            "annotated": annotated,
            **metrics,
        })

    # Mesure des performances
    print("\nMesure des performances...")
    timing = measure_performance(GROUND_TRUTH, resources, runs_per_query=10)
    for r in results:
        r["time_ms"] = timing.get(r["id"], 0.0)

    print_table(results)
    plot_results(results)


if __name__ == "__main__":
    main()
