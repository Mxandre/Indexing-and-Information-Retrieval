"""Construit l'anti-dictionnaire par lemmatisation spaCy et TF-IDF.

Variante de TD2/TD2.py qui lemmatise les tokens avant de calculer les scores
TF-IDF, afin d'obtenir un anti-dictionnaire basé sur les lemmes plutôt que
sur les formes fléchies.
"""

import re
from collections import Counter, defaultdict
from math import log
from pathlib import Path

import xml.etree.ElementTree as ET
import spacy


BASE_DIR = Path(__file__).resolve().parent
path_xml = BASE_DIR / '../TD2/corpus.xml'
path_root = BASE_DIR.parent / 'TD3'

# Seuil en dessous duquel un lemme est considéré comme bruit (anti-dictionnaire)
seuil = 0.0001


def segmente_lemma(path):
    """Découpe le corpus en lemmes par document.

    Analyse les champs `titre` et `texte` de chaque document XML avec spaCy
    et retourne un compteur de lemmes par numéro de bulletin.

    Args:
        path (Path | str): Chemin du fichier XML à traiter.

    Returns:
        defaultdict[str, Counter]: Dictionnaire ``{bulletin: Counter(lemmes)}``.
    """
    Dict = defaultdict(Counter)

    tree = ET.parse(path)
    root = tree.getroot()
    nlp = spacy.load("fr_core_news_sm")

    for doc in root.findall('document'):
        num = doc.find('bulletin').text
        titre = doc.find('titre').text
        texte = doc.find('texte').text
        doc_nlp = nlp(titre + ' ' + texte)
        lemmas = [
            token.lemma_.lower()
            for token in doc_nlp
            if not token.is_space and not token.is_punct
        ]
        Dict[num].update(Counter(lemmas))

    return Dict


def tf_idf(Dict):
    """Calcule le score TF-IDF maximal de chaque lemme sur l'ensemble du corpus.

    Utilise la formule logarithmique ``(1 + log10(tf)) * log10(N/df)`` et
    conserve le score le plus élevé observé parmi tous les documents.
    Écrit les résultats triés dans `tf-idf.txt`.

    Args:
        Dict (defaultdict[str, Counter]): Compteurs de lemmes par bulletin.

    Returns:
        dict[str, float]: Dictionnaire ``{lemme: score_tfidf_max}``.
    """
    idfs = {}
    for counter in Dict.values():
        for token in counter:
            idfs[token] = idfs.get(token, 0) + 1

    for num, counter in Dict.items():
        for token, count in counter.items():
            tf = 1 + log(count, 10)
            idf = log(len(Dict) / idfs[token], 10)
            Dict[num][token] = tf * idf

    tf_idf_dict = {}
    for num, counter in Dict.items():
        for token, score in counter.items():
            if token not in tf_idf_dict or score > tf_idf_dict[token]:
                tf_idf_dict[token] = score

    with open('tf-idf.txt', 'w', encoding='utf-8') as f:
        sorted_tokens = sorted(tf_idf_dict.items(), key=lambda item: item[1], reverse=True)
        for token, score in sorted_tokens:
            f.write(f"{token:<20} {score:<20.15f}\n")

    return tf_idf_dict


def anti_dict(tf_idf_dict):
    """Construit l'anti-dictionnaire à partir des lemmes de faible TF-IDF.

    Sélectionne les lemmes dont le score est inférieur ou égal à `seuil` et
    les écrit dans `anti_dict.txt`.

    Args:
        tf_idf_dict (dict[str, float]): Scores TF-IDF par lemme.

    Returns:
        dict[str, str]: Dictionnaire ``{lemme: ""}`` des lemmes à supprimer.
    """
    result = {}
    with open('anti_dict.txt', 'w', encoding='utf-8') as f:
        for token, score in tf_idf_dict.items():
            if score <= seuil:
                result[token] = ""
                f.write(f"{token}\t \"\"\n")
    return result


def substitue(texte, file_anti_dict):
    """Remplace les tokens du texte par leur substitut défini dans un fichier anti-dictionnaire.

    Args:
        texte (str): Texte à filtrer.
        file_anti_dict (str): Chemin relatif (depuis `path_root`) du fichier
            anti-dictionnaire au format TSV ``token<TAB>substitut``.

    Returns:
        str: Texte filtré, ou texte d'origine si vide/introuvable.
    """
    if not texte:
        return ""
    if not path_root + file_anti_dict:
        return texte

    subs_dict = {}
    with open(path_root + file_anti_dict, 'r', encoding='utf-8') as f:
        for line in f:
            token, subs = line.rstrip('\n').strip().split('\t')
            subs_dict[token] = subs.strip().strip('"')

    texte_filtre = texte
    for token, subs in subs_dict.items():
        texte_filtre = re.sub(
            r'\b' + re.escape(token) + r'\b', subs, texte_filtre, flags=re.IGNORECASE
        )
    return texte_filtre


def substitue_dict(texte, anti_dict):
    """Remplace les tokens du texte par leur substitut défini dans un dictionnaire.

    Args:
        texte (str): Texte à filtrer.
        anti_dict (dict[str, str]): Dictionnaire ``{token: substitut}``.

    Returns:
        str: Texte filtré, ou texte d'origine si vide/dict absent.
    """
    if not texte:
        return ""
    if not anti_dict:
        return texte

    texte_filtre = texte
    for token, subs in anti_dict.items():
        texte_filtre = re.sub(
            r'\b' + re.escape(token) + r'\b', subs, texte_filtre, flags=re.IGNORECASE
        )
    return texte_filtre


def creer_xml_filtre(xml_entree, xml_sortie, subst):
    """Crée un fichier XML filtré en substituant les tokens des champs textuels.

    Accepte en paramètre `subst` soit un chemin de fichier TSV, soit un
    dictionnaire de substitution directement.

    Args:
        xml_entree (str | Path): Chemin du fichier XML source.
        xml_sortie (str | Path): Chemin du fichier XML de sortie.
        subst (str | dict): Chemin du fichier TSV ou dictionnaire
            ``{token: substitut}``.
    """
    tree = ET.parse(xml_entree)
    root = tree.getroot()

    if isinstance(subst, str):
        subs_dict = {}
        with open(subst, 'r', encoding='utf-8') as f:
            for line in f:
                token, subs = line.rstrip('\n').strip().split('\t')
                subs_dict[token] = subs.strip().strip('"')
    else:
        subs_dict = subst

    for doc in root.findall('document'):
        titre = doc.find('titre')
        texte = doc.find('texte')
        if titre is not None:
            titre.text = substitue_dict(titre.text, subs_dict)
        if texte is not None:
            texte.text = substitue_dict(texte.text, subs_dict)

    tree.write(xml_sortie, encoding='utf-8', xml_declaration=True)


if __name__ == "__main__":
    Dict_lemma = segmente_lemma(path_xml)
    Dict_lemma_tfidf = tf_idf(Dict_lemma)
    anti_dict = anti_dict(Dict_lemma_tfidf)
    creer_xml_filtre(path_xml, 'corpus_filtre.xml', anti_dict)
