"""Lemmatisation Snowball et spaCy sur un corpus XML, avec statistiques de compression.

Compare la racinisation Snowball (NLTK) à la lemmatisation spaCy sur les textes
du corpus, puis affiche le taux de compression (lemmes uniques / formes totales).
"""

import xml.etree.ElementTree as ET
from collections import Counter

import nltk
import spacy
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


# Téléchargement des modèles de tokenisation NLTK si absents
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


def analyser_snowball(chemin_xml, chemin_sortie_csv):
    """Analyse un corpus XML avec le racineur Snowball français.

    Extrait les titres et textes de chaque document, tokenise avec NLTK,
    applique le racineur Snowball et écrit les paires (mot, racine) dans
    `mot_racine.txt`.

    Args:
        chemin_xml (str): Chemin vers le fichier XML à analyser.
        chemin_sortie_csv (str): Chemin vers le fichier de sortie (non utilisé
            directement ; l'écriture se fait dans `mot_racine.txt`).

    Returns:
        tuple[Counter, int]: Compteur des racines et nombre total de mots analysés.
    """
    stemmer = SnowballStemmer("french")

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

    texte_complet = " ".join(textes_extraits)
    mots = word_tokenize(texte_complet, language='french')

    stats_racines = Counter()
    nombre_total_mots = 0

    with open("mot_racine.txt", mode="w", encoding="utf-8", newline="") as f:
        for mot in mots:
            if mot.isalnum():
                mot_minuscule = mot.lower()
                racine_mot = stemmer.stem(mot_minuscule)
                f.write(f"{mot}\t{racine_mot}\n")
                stats_racines[racine_mot] += 1
                nombre_total_mots += 1

    return stats_racines, nombre_total_mots


def extract_mot_lemma_from_xml(xml_file):
    """Extrait les paires (forme, lemme) de tous les documents d'un corpus XML.

    Applique le pipeline spaCy sur les champs `texte` et `titre` de chaque
    document et ne conserve que les tokens alphabétiques.

    Args:
        xml_file (str): Chemin vers le fichier XML du corpus.

    Returns:
        list[tuple[str, str]]: Liste de paires ``(forme_originale, lemme)``.
    """
    nlp = spacy.load('fr_core_news_sm')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    mot_lemma_list = []

    for corpus in root.findall('.//document'):
        text = corpus.find('.//texte')
        if text is not None and text.text:
            mots = nlp(text.text)
            for mot in mots:
                if mot.is_alpha:
                    mot_lemma_list.append((mot.text, mot.lemma_))
        title = corpus.find('.//titre')
        if title is not None and title.text:
            mots = nlp(title.text)
            for mot in mots:
                if mot.is_alpha:
                    mot_lemma_list.append((mot.text, mot.lemma_))

    return mot_lemma_list


def calculate_lemma_frequencies(mot_lemma_list):
    """Calcule la fréquence de chaque lemme dans une liste de paires (forme, lemme).

    Args:
        mot_lemma_list (list[tuple[str, str]]): Liste de paires telle que
            retournée par `extract_mot_lemma_from_xml`.

    Returns:
        dict[tuple[str, str], int]: Fréquence de chaque paire (forme, lemme).
    """
    lemma_freq = {}
    for lemma in mot_lemma_list:
        if lemma in lemma_freq:
            lemma_freq[lemma] += 1
        else:
            lemma_freq[lemma] = 1
    return lemma_freq


if __name__ == "__main__":
    with open("TD3/lemma_frequencies.txt", 'r', encoding="utf-8") as f:
        length_freq = len([line for line in f if line.strip()])
        print(f"Nombre de lemmes uniques : {length_freq}")
    with open("TD3/mot_lemma_list.txt", 'r', encoding="utf-8") as f:
        length_all = len([line for line in f if line.strip()])
        print(f"Nombre total de formes : {length_all}")
    print(f"Taux de compression : {length_freq / length_all}")
