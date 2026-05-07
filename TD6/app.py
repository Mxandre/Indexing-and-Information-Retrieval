import sys
import pickle
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime
import xml.etree.ElementTree as ET

from flask import Flask, render_template, request
import spacy

# Configurations des chemins
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir / "TD5"))

import moteur

app = Flask(__name__)
nlp = spacy.load("fr_core_news_sm")

# Chargement global des modèles lexicaux pour accélérer les requêtes Web
print("Chargement des données du moteur...")
if not moteur.PICKLE_TF_IDF_FILE.exists() or not moteur.PICKLE_ANTI_LIST.exists():
    moteur.load_lemmatisatioin_file()

with open(moteur.PICKLE_TF_IDF_FILE, 'rb') as file:
    tf_idf_dict = pickle.load(file)
with open(moteur.PICKLE_ANTI_LIST, "rb") as file:
    anti_list = pickle.load(file)

index_inverse = moteur.charger_index(moteur.INDEX_INVERSE_FULL_FILE)

# Chargement du corpus XML en mémoire pour extraire les snippets et métadonnées
corpus_data = {}
CORPUS_FILE = parent_dir / "TD2" / "corpus.xml"

if CORPUS_FILE.exists():
    tree = ET.parse(CORPUS_FILE)
    for doc in tree.getroot().findall('document'):
        doc_id = doc.find('article').text.strip() if doc.find('article') is not None and doc.find('article').text else ""
        date_text = doc.find('date').text.strip() if doc.find('date') is not None and doc.find('date').text else ""
        rubrique = doc.find('rubrique').text.strip() if doc.find('rubrique') is not None and doc.find('rubrique').text else ""
        titre = doc.find('titre').text.strip() if doc.find('titre') is not None and doc.find('titre').text else ""
        texte = doc.find('texte').text.strip() if doc.find('texte') is not None and doc.find('texte').text else ""
        
        try:
            date_obj = datetime.strptime(date_text, "%d/%m/%Y")
        except ValueError:
            date_obj = datetime.min
            
        corpus_data[doc_id] = {
            'id': doc_id,
            'date': date_text,
            'date_obj': date_obj,
            'rubrique': rubrique,
            'titre': titre,
            'texte': texte
        }
else:
    print(f"Fichier corpus introuvable : {CORPUS_FILE}")

def get_highlight_keywords(query: str, anti_list: list) -> list:
    """Extrait tous les mots pertinents pour les surligner dans les snippets."""
    doc = nlp(query)
    keywords = []
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in anti_list and token.is_alpha:
            keywords.append(lemma)
            keywords.append(token.text.lower())
    return list(set(keywords))

def generate_snippet(texte: str, keywords: list, window: int = 80) -> str:
    """Génère un extrait de texte (snippet) centré sur le premier mot-clé trouvé."""
    if not keywords:
        return texte[:200] + "..."
    
    texte_lower = texte.lower()
    first_idx = -1
    
    # Recherche de la première occurrence d'un des mots clés
    for kw in keywords:
        idx = texte_lower.find(kw)
        if idx != -1:
            if first_idx == -1 or idx < first_idx:
                first_idx = idx
                
    if first_idx == -1:
        snippet = texte[:200] + "..."
    else:
        start = max(0, first_idx - window)
        end = min(len(texte), first_idx + window * 2)
        snippet = texte[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(texte):
            snippet = snippet + "..."
            
    # Surlignage avec la balise <mark>
    # On trie les keywords par longueur décroissante pour éviter qu'un mot court n'interrompe le surlignage d'un long
    for kw in sorted(keywords, key=len, reverse=True):
        snippet = re.sub(rf"({re.escape(kw)})", r"<mark>\1</mark>", snippet, flags=re.IGNORECASE)
        
    return snippet

@app.route("/", methods=["GET", "POST"])
def index():
    query = request.form.get("query", "")
    sort_by = request.form.get("sort", "relevance")
    results = []
    
    if query:
        # 1. Récupération des mots pour le surlignage et le score
        keywords = get_highlight_keywords(query, anti_list)
        
        # 2. Evaluation booléenne avec moteur.py pour obtenir les IDs
        doc_ids = moteur.evaluer_requete_recursive(query, tf_idf_dict, anti_list, index_inverse)
        
        # 3. Récupération du contexte dans le corpus original XML
        for doc_id in doc_ids:
            if doc_id in corpus_data:
                doc = corpus_data[doc_id]
                snippet = generate_snippet(doc['texte'], keywords)
                
                # Calcul de pertinence basique (fréquence des mots-clés dans le titre et le texte)
                score = 0
                for kw in keywords:
                    score += doc['titre'].lower().count(kw) * 2 # le titre a plus de poids
                    score += doc['texte'].lower().count(kw)
                
                result_item = doc.copy()
                result_item['snippet'] = snippet
                result_item['score'] = score
                results.append(result_item)
                
        # 4. Tri post-recherche
        if sort_by == "relevance":
            results.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == "date_asc":
            results.sort(key=lambda x: x['date_obj'])
        elif sort_by == "date_desc":
            results.sort(key=lambda x: x['date_obj'], reverse=True)
            
    return render_template("index.html", query=query, results=results, sort_by=sort_by)

if __name__ == "__main__":
    app.run(debug=True, port=5000)