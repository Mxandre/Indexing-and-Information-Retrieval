import xml.etree.ElementTree as ET
import spacy

def inverse_index(file_xml):
    tree = ET.parse(file_xml)
    root = tree.getroot()
    nlp = spacy.load('fr_core_news_sm')
    inverse_index = {}
    for doc in root.findall('document'):
        doc_id = doc.find('bulletin').text
        title = doc.find('titre').text
        text = doc.find('texte').text
        if text is not None and text.strip():
            doc_nlp = nlp(title + ' ' + text)
            doc_lemmas = [token.lemma_.lower() for token in doc_nlp if not token.is_space and not token.is_punct and token.is_alpha]
            for lemma in doc_lemmas:
                if lemma not in inverse_index:
                    inverse_index[lemma] = {}
                if doc_id not in inverse_index[lemma]:
                    inverse_index[lemma][doc_id] = 0
                inverse_index[lemma][doc_id] += 1
    return inverse_index

if __name__ == "__main__":
    file_xml = 'TD3/corpus_filtre_doub.xml'
    inverse_index = inverse_index(file_xml)
    with open('inverse_index.txt', 'w', encoding='utf-8') as f:
        for lemma, doc_dict in inverse_index.items():
            doc_freqs = ', '.join(f"{doc_id}: {freq}" for doc_id, freq in doc_dict.items())
            f.write(f"{lemma}\t{doc_freqs}\n") 