from bs4 import BeautifulSoup
from lxml import etree
import xml.etree.ElementTree as ET
import re, glob
from xml.dom import minidom


# Chemin complet vers le dossier contenant les fichiers HTM par défaut
my_path = '/Users/lipengcheng/Programming/py/LO17/BULLETINS/'

def get_path() :
    """Retourne le chemin du dossier contenant les fichiers HTM.

    Returns:
        str: Chemin du dossier.
    """
    read_path = input("Veuillez entrer le chemin du dossier contenant les fichiers HTM (ou appuyez sur Entrée pour utiliser le chemin par défaut) : ")
    if read_path.strip() != "" :
        if glob.glob(read_path + "/*.htm") :
            return read_path.strip()
        else :
            print("Le chemin spécifié est invalide ou ne contient pas de fichiers .htm. Utilisation du chemin par défaut.")
            return my_path

def extraire_infos(contenu) :
    """Extrait les metadonnees d'un article depuis son HTML.

    Cette fonction analyse le contenu avec BeautifulSoup et retourne un
    dictionnaire contenant les champs principaux de l'article (bulletin, date,
    rubrique, titre, auteur, texte, images, contact).

    Args:
        contenu (str): Contenu HTML brut du fichier article.

    Returns:
        dict: Dictionnaire des informations extraites.
    """

    # initialiser le dictionnaire pour stocker les informations extraites
    infos = {
      "article" : "",
      "bulletin" : "",
      "date" : "",
      "rubrique" : "",
      "titre" : "",
      "auteur" : "",
      "texte" : "",
      "images" : [],
      "contact" : "",
    }

    # parser le contenu HTML avec BeautifulSoup
    soup = BeautifulSoup(contenu, "html.parser")
    title_tag = soup.find('title')
    if title_tag:
        # Le titre contient "BE France 287"
        match = re.search(r'BE\s+\w+\s+(\d+)', title_tag.get_text())
        if match:
            infos['bulletin'] = match.group(1)
    for span in soup.find_all('span', class_ = "style15"):
        text_origin = span.get_text()
        match_art = re.search(r'/(\d+)\.htm', text_origin)
        if match_art :
            infos["article"] = match_art.group(1)
        match_date = re.search(r'(\d{1,2}/\d{2}/\d{4})', text_origin)
        if match_date :
            infos["date"] = match_date.group(1)


    ## rubrique   style 42
    for ele in soup.find_all('span', class_ = "style42") :
        text_origin = ele.get_text(strip = True)
        if not re.match(r'\d+/\d+/\d+', text_origin) :
            infos["rubrique"] = text_origin
            break
        
    ## titre   style 17
    titre_tag = soup.find('span', class_ = "style17")
    if titre_tag :
        infos["titre"] = titre_tag.get_text(strip = True)


    ## auteur
    ## d'abord trouver le td qui contient "dacteur" puis trouver le td suivant pour extraire le nom de l'auteur
    for td in soup.find_all('td') :
        span = td.find('span', class_ = "style28")
        if span and "dacteur" in span.get_text() :    
            next_td = td.find_next_sibling('td')
            if next_td :
                match = re.search(r'-\s*(.+?)-\s*email', next_td.get_text())
                if match :
                    infos["auteur"] = match.group(1).strip()
                    break


    ## contact
    for td in soup.find_all('td') :
        span = td.find('span', class_ = "style28")
        if span and "contact" in span.get_text() :
            next_td = td.find_next_sibling('td')
            if next_td :
                infos["contact"] = next_td.get_text(strip = True)

    ## image_src et image_legend

    for img_tag in soup.find_all("img", style = lambda s : s and "margin-bottom:5px"  in s) :
        img_src = img_tag.get('src')
        parent = img_tag.parent
        if parent and parent.name == "a" :
            find_node = parent
        else:
            find_node = img_tag
        br_legend = find_node.find_next_sibling('br')
        if br_legend :
            span_legend = br_legend.find('span', class_ = "style21")
            if span_legend :
                infos["images"].append((img_src,span_legend.get_text(strip = True)))
        else :
            infos["images_legend"].append("")
    
    return infos


def titre_parse(file_content):
    """Extrait le titre de l'article via XPath.

    Args:
        file_content (str): Contenu HTML brut du fichier article.

    Returns:
        str: Titre de l'article.
    """
    
    parser = etree.HTMLParser()
    tree = etree.fromstring(file_content, parser)

    #titre se trouve dans un span de class style17
    titre = tree.xpath("//span[@class='style17']/text()")
    return titre[0]


def num_article_parse(file_content):
    """Extrait le numero d'article a partir de l'URL presente dans le HTML.

    Args:
        file_content (str): Contenu HTML brut du fichier article.

    Returns:
        str | None: Numero d'article extrait, ou `None` si non trouve.
    """
    parser = etree.HTMLParser()
    tree = etree.fromstring(file_content, parser)

    #numero d'article se trouve dans un span de class style88
    num_article_raw = tree.xpath("//span[@class='style88']/text()")
    match = re.search(r'/(\d+)\.htm$',num_article_raw[0])
    if match:
        return match.group(1)
    
def texte_parse(file_content):
    """Extrait et nettoie le texte de l'article via XPath.

    Args:
        file_content (str): Contenu HTML brut du fichier `.htm`.

    Returns:
        str: Texte de l'article concatene sans sauts de ligne parasites.
    """
    parser = etree.HTMLParser()
    tree = etree.fromstring(file_content, parser)

    texte = tree.xpath("//p[@class='style96']//span[@class='style95']/text()")
    
    # supprimer les sauts de ligne, concaténer les paragraphes
    return "".join([para for para in texte if para != "\n"])  




def Creer_Corpus(archive_path):
    """Construit un corpus XML a partir des fichiers HTML d'un dossier.

    La fonction parcourt tous les `.htm`, extrait les informations avec bs4 et
    XPath (En raison du travail en binôme, différentes méthodes ont été utilisées.), 
    puis ecrit le resultat agrege dans `corpus.xml`.

    Args:
        archive_path (str): Chemin du dossier contenant les fichiers `.htm`.

    Returns:
        None: Cette fonction ecrit le fichier XML sur disque.
    """

    # créer la structure XML de base
    corpus_root = ET.Element("corpus")

    # parcourir tous les fichiers .htm dans le dossier
    for file_path in glob.glob(archive_path + "/*.htm"):
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # créer le nœud XML pour le document
        doc = ET.SubElement(corpus_root, "document")

        infos = extraire_infos(html_content)

         # numero d'article
        ET.SubElement(doc, "article").text = num_article_parse(html_content)

        # numéro de bulletin
        ET.SubElement(doc, "bulletin").text = infos["bulletin"]

        # date
        ET.SubElement(doc, "date").text = infos["date"]

        #Rubrique
        ET.SubElement(doc, "rubrique").text = infos["rubrique"]

        # titre
        ET.SubElement(doc, "titre").text =  titre_parse(html_content)

        # auteur
        ET.SubElement(doc, "auteur").text = infos["auteur"]

        # texte
        ET.SubElement(doc, "texte").text = texte_parse(html_content)

        # images
        images_element = ET.SubElement(doc, "images")
        for img_url, legend in infos["images"] :
            ET.SubElement(images_element, "image").text = img_url
            ET.SubElement(images_element, "legend").text = legend

        # contact
        ET.SubElement(doc, "contact").text = infos["contact"]

    xml_str = ET.tostring(corpus_root, encoding='utf-8')

    #formater le XML avec indentation 
    dom = minidom.parseString(xml_str)
    xml_indent = dom.toprettyxml(indent="    ")  # 4 espaces d'indentation
    with open("corpus.xml", "w", encoding="utf-8") as f:
        f.write(xml_indent)


# --- Exécution Principale ---
if __name__ == "__main__":
    
    # Création du corpus XML à partir de tous les fichiers
    Creer_Corpus(get_path())
