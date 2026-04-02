import re
from pathlib import Path
from collections import defaultdict

DMY_PATTERN_NUMERO = re.compile(r"\b(?P<day>\d{1,2})[/\-\s]+(?P<month>\d{1,2})[/\-\s]+(?P<year>\d{4})\b", re.IGNORECASE)
DMY_PATTERN_TEXTE = re.compile(r'\b(?P<day>\d{1,2})[/\-\s]+(?P<month_name>janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)[/\-\s]+(?P<year>\d{4})\b', re.IGNORECASE)
MY_PATTERN = re.compile(r"\b(?P<month>\d{1,2})[/\-\s]+(?P<year>\d{4})\b", re.IGNORECASE)
MY_TEXT = re.compile(r"\b(?P<month_name>janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)[\-\s]+(?P<year>\d{4})\b", re.IGNORECASE)
Y_PATTERN = re.compile(r"\b(?P<year>19\d{2}|20\d{2})\b", re.IGNORECASE)
RUBRIQUE_PATTERN = re.compile(r"\brubrique\s+(?P<rubrique>[A-Za-zÀ-ÿ\-]+)\b", re.IGNORECASE)

OP_PATTERN = re.compile(r"(?P<part1>.*)\b(et|ou|sans|)(?P<part2>.*)", re.IGNORECASE)

PATTERNS = {
    "dmy_numero": DMY_PATTERN_NUMERO,
    "dmy_text": DMY_PATTERN_TEXTE,
    "my_numero": MY_PATTERN,
    "my_text": MY_TEXT,
    "y": Y_PATTERN,
    "rubrique": RUBRIQUE_PATTERN,
    "op": OP_PATTERN
}


def pipeline_traitement_requete(source : str, metadonnees : dict) -> dict:
    metadonnees = traiter_metadonnees(source, metadonnees)
    metadonnees = traiter_op_logique(source, metadonnees)
    metadonnees = traiter_mots_cles(source, metadonnees)
    return metadonnees





def traiter_metadonnees(source : str, metadonnees : defaultdict) -> dict:
    '''
    Traitez les métadonnées ici, incluant les Les contraintes temporelles, Les rubriques spécifiques de l’ADIT,
    
    Les filtres structurels
    Args:
        source : la requete a traiter
        metadonnees : dictionnaire contenant les métadonnées extraites de la requete

    Returns:
        dict: un dictionnaire contenant les métadonnées traitées
    '''
    dmy_numero = PATTERNS["dmy_numero"].search(source)
    dmy_texte = PATTERNS["dmy_text"].search(source)
    if dmy_numero is not None or dmy_texte is not None:
        metadonnees["date"] = {dmy_numero if dmy_numero else dmy_texte}
    else : 
        my_numero = PATTERNS["my_numero"].search(source)
        my_texte = PATTERNS["my_text"].search(source)
        if my_numero is not None or my_texte is not None:
            metadonnees["date"] = {my_numero if my_numero else my_texte}
        else : 
            y_pattern = PATTERNS["y"].search(source)
            if y_pattern:
                metadonnees["date"] = {y_pattern}
    rubrique = PATTERNS["rubrique"].search(source)
    if rubrique:
        metadonnees["rubrique"] = {rubrique.group("rubrique")}
    
    return metadonnees  # Retournez les métadonnées traitées



def traiter_op_logique(source : str, metadonnees : dict) -> list:

    '''
    Traitez les opérateurs logiques ici, incluant les opérateurs logiques (AND, OR, NOT)

    Args:
        source : la requete a traiter
        metadonnees : dictionnaire contenant les métadonnées extraites de la requete

    Returns:
        dict: un dictionnaire contenant les métadonnées traitées
    '''

    if PATTERNS["op"].search(source):
        op_match = PATTERNS["op"].search(source)
        part1 = op_match.group("part1").strip()
        metadonnees = traiter_op_logique(part1, metadonnees)
        part2 = op_match.group("part2").strip()
        metadonnees = traiter_op_logique(part2, metadonnees)
    else :
        traiter_op_logique["parts"].append(source.strip())

    return metadonnees  

def traiter_mots_cles(source : str, metadonnees : dict) -> dict:
    '''
    Traitez les mots-clés restant dans la requete ici

    Args:
        source : la requete a traiter
        metadonnees : dictionnaire contenant les métadonnées extraites de la requete

    Returns:
        dict: un dictionnaire contenant les métadonnées traitées
    '''
    return metadonnees  

def extraire_request(file_path:Path)-> str :
    with open(file_path, 'r', encoding = 'utf-8') as f:
        contenu = f.read()
    contenu = re.sub(r"\s+", " ", contenu)
    morceaus = contenu.split("?")
    requests = []
    for morceau in morceaus:
        petit_morceaus  = morceau.split(".")
        for petit_morceau in petit_morceaus:
            req = petit_morceau.strip()
            req = re.sub(r"^\—\s*", "", req)
            if req:
                requests.append(req)
    return requests



if __name__ == "__main__":
    ## pretraitement de text
    file_name = Path(__file__).parent/"requete.txt"
    # with open(file_name, 'w', encoding = 'utf-8') as f:
    #    requests = extraire_request(Path(__file__).parent/"test.txt")
    #    f.write("\n".join(requests))
    
    ## test the date and op
    with open(file_name,'r', encoding="utf-8") as f:
        requests = f.readlines()
    for request in requests :
        metadonnes = defaultdict(list)
        metadonnes = traiter_metadonnees(request, metadonnes)
        if len(metadonnes.keys()) >=1 :
            print(request)
            print("the result obtained is ", metadonnes)
            
