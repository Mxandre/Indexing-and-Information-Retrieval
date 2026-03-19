import xml.etree.ElementTree as ET
import spacy

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
