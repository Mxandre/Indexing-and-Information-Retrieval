import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import re, glob
from math import log
import spacy

'''
Ce TD consacré à determination de l'anti-dictionnaire basé sur les fichiers HTML 
et à la construction d'un corpus XML structuré.
'''

path_xml = 'TD2/corpus.xml'
path_root = '/Users/lipengcheng/Programming/py/LO17/'

# seuil pour filtrer les tokens à inclure dans l'anti-dictionnaire
seuil = 0.0001

def segmente(path) : 
    '''
    découpe le corpus (les titres et les textes) en tokens.
    '''
    # utilisant un dictionnaire embarqué pour stocker les tokens dans chaque document, 
    # et leurs fréquences pour chaque document
    Dict = defaultdict(Counter)

    tree = ET.parse(path)
    root = tree.getroot()
    nlp = spacy.load("fr_core_news_sm")

    for doc in root.findall('document'):

        num = doc.find('bulletin').text
        titre = doc.find('titre').text
        texte = doc.find('texte').text
        doc_nlp = nlp(titre + ' ' + texte)
        tokens = [token.text.lower() for token in doc_nlp if not token.is_space and not token.is_punct]
        mot_lemma_dict = {}
        with open("TD3/mot_lemma_list.txt", 'r', encoding = "utf-8") as f:
            for line in f :
                mot, lemma = line.strip().split('\t')
                mot_lemma_dict[mot] = lemma
        tokens = [mot_lemma_dict.get(token, token) for token in tokens]
        
        #tokens = re.findall(r'\b\w+\b', titre + ' ' + texte)

        Dict[num].update(Counter(tokens))
    return Dict



def tf_idf(Dict):   
    '''
    calcule le score TF-IDF pour chaque token dans chaque document 
    et puis garde le score maximum de chaque token à travers tous les documents,
    
    retourne un dictionnaire de tokens avec leurs scores TF-IDF.
    '''
    N = len(Dict)
    idfs = {}

    for counter in Dict.values():
        for token in counter:
            idfs[token] = idfs.get(token, 0) + 1

    for num, counter in Dict.items():
        for token, count in counter.items():
            tf = count
            idf = log(len(Dict) / idfs[token])
            tf_idf_score = tf * idf
            Dict[num][token] = tf_idf_score
    
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
    anti_dict = {}
    with open('anti_dict.txt', 'w', encoding='utf-8') as f:
        for token, score in tf_idf_dict.items():
            if score <= seuil:
                    anti_dict[token] = ""
                    f.write(f"{token}\t \"\"\n")
    return anti_dict

def substitue(texte, file_anti_dict):
    '''
    substitue les tokens du texte par les tokens de la fichier anti-dictionnaire(Fichier txt).
    '''
    if not texte:
        return ""
    if not path_root+file_anti_dict:
        return texte

    # lire le fichier anti-dictionnaire et construire un dictionnaire de substitution
    subs_dict = {}
    with open(path_root+file_anti_dict, 'r', encoding='utf-8') as f:
        for line in f:
            token, subs = line.rstrip('\n').strip().split('\t')
            subs_dict[token] = subs.strip().strip('"')  

    texte_filtre = texte

    for token, subs in subs_dict.items():
        texte_filtre = re.sub(r'\b' + re.escape(token) + r'\b', subs, texte_filtre, flags=re.IGNORECASE)

    return texte_filtre

def substitue_dict(texte, anti_dict):
    '''
    substitue les tokens du texte par les tokens de la anti-dictionnaire(Dict).
    '''

    if not texte:
        return ""
    if not anti_dict:
        return texte

    texte_filtre = texte

    for token, subs in anti_dict.items():
        texte_filtre = re.sub(r'\b' + re.escape(token) + r'\b', subs, texte_filtre, flags=re.IGNORECASE)

    return texte_filtre

def creer_xml_filtre(xml_entree, xml_sortie, subst):
    '''
    crée un nouveau fichier XML filtré en substituant les tokens du texte
    '''

    tree = ET.parse(xml_entree)
    root = tree.getroot()

    # si subst est le chemin d'un fichier, lire le fichier et construire un dictionnaire de substitution
    if isinstance(subst, str):
        fichier_subst = subst
        subs_dict = {}
        with open(fichier_subst, 'r', encoding='utf-8') as f:
            for line in f:
                token, subs = line.rstrip('\n').strip().split('\t')
                subs_dict[token] = subs.strip().strip('"')  
    else:
        subs_dict = subst
    
    
    for doc in root.findall('document'):
        titre = doc.find('titre')
        texte = doc.find('texte')
        if titre is not None: titre.text = substitue_dict(titre.text, subs_dict)
        if texte is not None: texte.text = substitue_dict(texte.text, subs_dict)
        
    tree.write(xml_sortie, encoding='utf-8', xml_declaration=True)


if __name__ == "__main__":
    Dict = segmente(path_xml)
    Dict = tf_idf(Dict)
    anti_dict = anti_dict(Dict)
    creer_xml_filtre(path_xml, 'corpus_filtre.xml', anti_dict)
   