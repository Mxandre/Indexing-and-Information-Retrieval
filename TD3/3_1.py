from pydoc import doc, text
import xml.etree.ElementTree as ET
import spacy
from collections import defaultdict
import re

def create_inverse_index(file_xml):
    tree = ET.parse(file_xml)
    root = tree.getroot()
    nlp = spacy.load('fr_core_news_sm')
    # Dictionnaire embarqué pour stocker l'index inverse par zones
    # Structure : {lemma: {doc_id: {zone: frequency}}}
    inverse_index = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for doc in root.findall('document'):
        doc_id = doc.find('bulletin').text.strip()

        # zones : faut faire nlp
        zones = ['titre', 'texte']
        for zone in zones:
            node = doc.find(zone)
            if node is not None and node.text and node.text.strip():
                content = node.text
                text_nlp = nlp(content)
                text_lemmas = [token.lemma_.lower() for token in text_nlp if not token.is_space and not token.is_punct and token.is_alpha]
                for lemma in text_lemmas:
                    inverse_index[lemma][doc_id][zone] += 1
                    
        # champs, metadonnees : pas de nlp, juste zone exacte
        champs = ['auteur', 'date', 'rubrique', 'article']
        for champ in champs:
            node = doc.find(champ)
            if node is not None and node.text and node.text.strip():
                content = node.text
                # exact zone
                inverse_index[content.lower().strip()][doc_id][champ] += 1

        # presence des images : Index particulier pour la existence d'une image dans le document
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