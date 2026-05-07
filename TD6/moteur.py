import sys
import pickle
from pathlib import Path
from collections import defaultdict
import ast

# Ajout du répertoire TD5 au path pour pouvoir importer traitement_requete
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir / "TD5"))

from traitement_requete import (
    normaliser_texte,
    pipeline_traitement_requete,
    PICKLE_TF_IDF_FILE,
    PICKLE_ANTI_LIST,
    load_lemmatisatioin_file
)

# Définition des chemins vers vos fichiers inverses générés au TD3.
INDEX_INVERSE_FULL_FILE = parent_dir / "TD3" / "inverse_index_full.txt"

def charger_index(filepath: Path) -> dict:
    """Charge un fichier inverse textuel (ou pickle)."""
    if filepath.exists():
        index = {}
        try:
            # Lecture du format spécifique généré par 3_1.py:
            # lemma \t doc_id.zone: freq, doc_id.zone: freq
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        lemma = parts[0]
                        # Extraire les IDs de documents uniques (avant le '.')
                        doc_ids = {entry.split('.')[0] for entry in parts[1].split(', ') if '.' in entry}
                        index[lemma] = doc_ids
            return index
        except Exception:
            # Fallback si c'est un pickle binaire
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    else:
        print(f"Attention: le fichier {filepath.name} est introuvable. Index vide utilisé.")
        return {}

def evaluer_metadonnees(metadonnees: dict, index_inverse: dict) -> set:
    """Recherche les doc_ids pour les mots-clés, la rubrique et la date dans l'index unifié, puis fait un ET (intersection)."""
    
    # 1. Évaluation des mots clés
    mots = metadonnees.get("key_word", [])
    resultats_mots = None
    if mots:
        for mot in mots:
            docs = index_inverse.get(mot, set())
            if resultats_mots is None:
                resultats_mots = set(docs)
            else:
                resultats_mots = resultats_mots.intersection(docs)
    else:
        resultats_mots = set() # Vide si aucun mot pertinent

    # 2. Évaluation de la rubrique
    resultats_rubrique = None
    if "rubrique" in metadonnees:
        rub = metadonnees["rubrique"].lower()
        resultats_rubrique = set(index_inverse.get(rub, set()))

    # 3. Évaluation de la date
    resultats_date = None
    if "date" in metadonnees:
        date_info = metadonnees["date"]
        resultats_date = set()
        if date_info["type"] in ("dmy", "my", "y"):
            resultats_date = set(index_inverse.get(date_info["value"], set()))
        elif date_info["type"] in ("between_dmy", "between_my", "between_y"):
            start, end = date_info.get("start", ""), date_info.get("end", "")
            # Recherche par intervalle sur les clés de l'index des dates
            for d_key, docs in index_inverse.items():
                if start <= str(d_key) <= end:
                    resultats_date.update(docs)

    # 4. Combinaison logique globale (Modèle Booléen strict: ET)
    resultats_finaux = resultats_mots if resultats_mots is not None else set()
    
    if resultats_rubrique is not None:
        resultats_finaux = resultats_finaux.intersection(resultats_rubrique) if resultats_mots is not None else resultats_rubrique
            
    if resultats_date is not None:
        if resultats_mots is None and resultats_rubrique is None:
            resultats_finaux = resultats_date
        else:
            resultats_finaux = resultats_finaux.intersection(resultats_date)

    return resultats_finaux if resultats_finaux else set()

def evaluer_requete_recursive(requete_texte: str, tf_idf_dict: dict, anti_list: list, index_inverse: dict) -> set:
    """Traite la requête en évaluant récursivement les opérateurs logiques (ET, OU, SANS)."""
    req_norm, upper_key_word = normaliser_texte(requete_texte, key_word_traite=True)
    meta = defaultdict(list)
    meta = pipeline_traitement_requete(req_norm, meta, tf_idf_dict, anti_list,upper_key_word)
    
    # Gestion de l'arbre booléen en utilisant l'opérateur trouvé dans le TD5
    if "operateur" in meta and "part1" in meta and "part2" in meta:
        op = meta["operateur"].lower()
        res_part1 = evaluer_requete_recursive(meta["part1"], tf_idf_dict, anti_list, index_inverse)
        res_part2 = evaluer_requete_recursive(meta["part2"], tf_idf_dict, anti_list, index_inverse)
        
        if op == "et":
            return res_part1.intersection(res_part2)
        elif op == "ou":
            return res_part1.union(res_part2)
        elif op == "sans":
            return res_part1.difference(res_part2)

    # S'il n'y a pas/plus d'opérateur, on évalue simplement le nœud final
    return evaluer_metadonnees(meta, index_inverse)


def lancer_moteur() :
    print("==================================================")
    print("          Lancement du Moteur de Recherche        ")
    print("==================================================")
    
    # 1. Vérification et chargement des modèles lexicaux (TF-IDF & Anti-dict)
    if not PICKLE_TF_IDF_FILE.exists() or not PICKLE_ANTI_LIST.exists():
        print("Génération des fichiers TF-IDF et Anti-dictionnaire manquants...")
        load_lemmatisatioin_file()

    with open(PICKLE_TF_IDF_FILE, 'rb') as file:
        tf_idf_dict = pickle.load(file)
    with open(PICKLE_ANTI_LIST, "rb") as file:
        anti_list = pickle.load(file)

    # 2. Chargement des fichiers inverses (TD3)
    print("Chargement des fichiers d'indexation inversée en cours...")
    index_inverse = charger_index(INDEX_INVERSE_FULL_FILE)
    
    print("Moteur de recherche prêt !\n")
    
    # 3. Boucle principale d'interaction avec l'utilisateur
    while True:
        requete = input("Entrez votre requête (ou 'quitter' pour arrêter) : ")
        if requete.strip().lower() in ['quitter', 'exit', 'q']:
            print("Arrêt du moteur...")
            break
            
        if not requete.strip():
            continue

        # 4 & 5. Traduction structurée, évaluation booléenne et récupération des ids
        doc_ids = evaluer_requete_recursive(
            requete, 
            tf_idf_dict, 
            anti_list, 
            index_inverse
        )
        
        # Affichage du résultat
        print(f"\n=> {len(doc_ids)} document(s) trouvé(s).")
        if doc_ids:
            # On trie les résultats pour un affichage plus propre
            print("ID(s) des documents :", ", ".join(map(str, sorted(list(doc_ids)))))
        print("-" * 50)

if __name__ == "__main__":
    lancer_moteur()