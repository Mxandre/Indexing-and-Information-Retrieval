"""Construit un index inverse multi-zones à partir d'un corpus XML filtré.

L'index résultant associe chaque lemme à l'ensemble des documents où il
apparaît, avec la fréquence par zone (`titre`, `texte`, `auteur`, `date`,
`rubrique`, `article`). Une entrée spéciale `has_image` indique la présence
d'images dans un document.
"""

import xml.etree.ElementTree as ET
import re
from collections import defaultdict

import spacy


def create_inverse_index(file_xml):
    """Construit l'index inverse multi-zones d'un corpus XML.

    Pour les zones de contenu libre (`titre`, `texte`), applique spaCy pour
    lemmatiser et ne conserver que les tokens alphabétiques non vides. Pour
    les métadonnées (`auteur`, `date`, `rubrique`, `article`), indexe la
    valeur exacte en minuscules.

    Args:
        file_xml (str): Chemin vers le fichier XML du corpus filtré.

    Returns:
        defaultdict: Index inverse de structure
        ``{lemme: {doc_id: {zone: fréquence}}}``.
    """
    tree = ET.parse(file_xml)
    root = tree.getroot()
    nlp = spacy.load('fr_core_news_sm')
    # Structure : {lemme: {doc_id: {zone: fréquence}}}
    inverse_index = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for doc in root.findall('document'):
        doc_id = doc.find('bulletin').text.strip()

        # Zones à lemmatiser
        for zone in ('titre', 'texte'):
            node = doc.find(zone)
            if node is not None and node.text and node.text.strip():
                text_nlp = nlp(node.text)
                lemmas = [
                    token.lemma_.lower()
                    for token in text_nlp
                    if not token.is_space and not token.is_punct and token.is_alpha
                ]
                for lemma in lemmas:
                    inverse_index[lemma][doc_id][zone] += 1

        # Métadonnées : indexation exacte de la valeur brute
        for champ in ('auteur', 'date', 'rubrique', 'article'):
            node = doc.find(champ)
            if node is not None and node.text and node.text.strip():
                content = node.text.lower().strip()
                inverse_index[content][doc_id][champ] += 1

        # Présence d'images : entrée booléenne dans l'index
        if doc.find('images') is not None:
            inverse_index['has_image'][doc_id]['image'] += 1

    return inverse_index


if __name__ == "__main__":
    file_xml = 'TD3/corpus_filtre_doub.xml'
    inverse_index = create_inverse_index(file_xml)
    with open('inverse_index_full.txt', 'w', encoding='utf-8') as f:
        for lemma, doc_dict in inverse_index.items():
            doc_freqs_list = []
            for doc_id, zone_dict in doc_dict.items():
                for zone, freq in zone_dict.items():
                    doc_freqs_list.append(f"{doc_id}.{zone}: {freq}")
            doc_freqs_str = ', '.join(doc_freqs_list)
            f.write(f"{lemma}\t{doc_freqs_str}\n")
