import xml.etree.ElementTree as ET
import spacy
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict

# Téléchargement du modèle de tokenisation de NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def analyser_snowball(chemin_xml, chemin_sortie_csv):
    """
    Analyse un fichier XML en utilisant le Snowball Stemmer pour extraire les racines des mots.
    Les résultats sont écrits dans un fichier txt avec deux colonnes : "Mot" et "Racine".
    Args:
        chemin_xml (str): Le chemin vers le fichier XML à analyser.
        chemin_sortie_csv (str): Le chemin vers le fichier CSV de sortie.
    Returns:
        stats_racines (Counter): Un compteur des racines de mots et de leur fréquence.
        nombre_total_mots (int): Le nombre total de mots analysés.
    """
    stemmer = SnowballStemmer("french")
    
    # Analyse du fichier XML et extraction des textes
    try:
        arbre = ET.parse(chemin_xml)
        racine = arbre.getroot()
    except Exception as e:
        print(f"Erreur XML : {e}")
        return Counter(), 0

    textes_extraits = []
    for doc in racine.findall('document'):
        titre = getattr(doc.find('titre'), 'text', '') or ""
        texte = getattr(doc.find('texte'), 'text', '') or ""
        textes_extraits.append(titre + " " + texte)

    # Tokenisation des textes 
    texte_complet = " ".join(textes_extraits)
    mots = word_tokenize(texte_complet, language='french')

    # Statistiques des racines et nombre total de mots
    stats_racines = Counter()
    nombre_total_mots = 0

    with open("mot_racine.txt", mode="w", encoding="utf-8", newline="") as f:
        for mot in mots:
            if mot.isalnum():
                mot_minuscule = mot.lower()
                racine_mot = stemmer.stem(mot_minuscule)
                
                # Écriture dans le fichier de sortie
                f.write(f"{mot}\t{racine_mot}\n")
                
                # Mise à jour des statistiques
                stats_racines[racine_mot] += 1
                nombre_total_mots += 1

    return stats_racines, nombre_total_mots


def extract_mot_lemma_from_xml(xml_file):
    nlp = spacy.load('fr_core_news_sm')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    mot_lemma_list = []
    for corpus in root.findall('.//document'):
        text = corpus.find('.//texte')
        if text is not None and text.text :
            mots = nlp(text.text)
            for mot in mots:
                if mot.is_alpha:
                    mot_lemma_list.append((mot.text, mot.lemma_))
        title = corpus.find('.//titre')
        if title is not None and title.text :
            mots = nlp(title.text)
            for mot in mots:
                if mot.is_alpha:
                    mot_lemma_list.append((mot.text, mot.lemma_))
    
    return mot_lemma_list

def calculate_lemma_frequencies(mot_lemma_list):
    lemma_freq = {}
    for lemma in mot_lemma_list:
        if lemma in lemma_freq:
            lemma_freq[lemma] += 1
        else:
            lemma_freq[lemma] = 1
    return lemma_freq

if __name__ == "__main__" :

    # xml_file = "TD3\corpus_filtre.xml"
    # mot_lemma_list = extract_mot_lemma_from_xml(xml_file)
    # with open("TD3/mot_lemma_list.txt", 'w', encoding = "utf-8") as f :
    #     for mot, lemma in mot_lemma_list :
    #         f.write(f"{mot}\t{lemma}\n")
    
    # with open("TD3/mot_lemma_list.txt", 'r', encoding = "utf-8") as f :
    #     mot_lemma_list = [line.strip().split('\t')[-1] for line in f if line.strip()]
    # lemma_freq = calculate_lemma_frequencies(mot_lemma_list)
    # sorted_lemma_freq = sorted(lemma_freq.items(), key = lambda x : x[1], reverse = True)
    # with open("TD3/lemma_frequencies.txt", 'w', encoding = "utf-8") as f :
    #     for lemma, freq in sorted_lemma_freq :
    #         f.write(f"{lemma}\t{freq}\n")

    with open("TD3/lemma_frequencies.txt", 'r', encoding = "utf-8") as f :
        length_freq = len([line for line in f if line.strip()])
        print(f"Number of unique lemmas :", length_freq)
    with open("TD3/mot_lemma_list.txt", 'r', encoding = "utf-8") as f :
        length_all = len([line for line in f if line.strip()])
        print(f"Number of all words :", length_all)
    print(f"Compress rate :", length_freq / length_all)
