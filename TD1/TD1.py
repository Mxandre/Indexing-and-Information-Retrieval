"""Construit un corpus XML à partir des fichiers HTML des bulletins scientifiques.

Parcourt un dossier de fichiers `.htm`, extrait les métadonnées (bulletin, date,
rubrique, titre, auteur, texte, images, contact) via BeautifulSoup et lxml/XPath,
puis écrit le résultat dans `corpus.xml`.
"""

import glob
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from bs4 import BeautifulSoup
from lxml import etree


# Chemin par défaut vers le dossier contenant les fichiers HTM
TD1_DIR = Path(__file__).resolve().parent
my_path = TD1_DIR / 'BULLETINS'



def get_path():
    """Demande le chemin du dossier HTM et vérifie son contenu.

    Returns:
        str: Chemin valide contenant des fichiers `.htm`, ou chemin par défaut.
    """
    read_path = input(
        "Veuillez entrer le chemin du dossier contenant les fichiers HTM"
        " (ou appuyez sur Entrée pour utiliser le chemin par défaut) : "
    )
    if read_path.strip() != "":
        if glob.glob(read_path + "/*.htm"):
            return read_path.strip()
        else:
            print(
                "Le chemin spécifié est invalide ou ne contient pas de fichiers .htm."
                " Utilisation du chemin par défaut."
            )
            return str(my_path)
    return str(my_path)


def extraire_infos(contenu):
    """Extrait les métadonnées d'un article depuis son HTML.

    Analyse le contenu avec BeautifulSoup et retourne un dictionnaire contenant
    les champs principaux de l'article (bulletin, date, rubrique, titre, auteur,
    texte, images, contact).

    Args:
        contenu (str): Contenu HTML brut du fichier article.

    Returns:
        dict: Dictionnaire des informations extraites.
    """
    infos = {
        "article": "",
        "bulletin": "",
        "date": "",
        "rubrique": "",
        "titre": "",
        "auteur": "",
        "texte": "",
        "images": [],
        "contact": "",
    }

    soup = BeautifulSoup(contenu, "html.parser")
    title_tag = soup.find('title')
    if title_tag:
        # Le titre contient « BE France 287 »
        match = re.search(r'BE\s+\w+\s+(\d+)', title_tag.get_text())
        if match:
            infos['bulletin'] = match.group(1)

    for span in soup.find_all('span', class_="style15"):
        text_origin = span.get_text()
        match_art = re.search(r'/(\d+)\.htm', text_origin)
        if match_art:
            infos["article"] = match_art.group(1)
        match_date = re.search(r'(\d{1,2}/\d{2}/\d{4})', text_origin)
        if match_date:
            infos["date"] = match_date.group(1)

    # Rubrique — classe style42
    for ele in soup.find_all('span', class_="style42"):
        text_origin = ele.get_text(strip=True)
        if not re.match(r'\d+/\d+/\d+', text_origin):
            infos["rubrique"] = text_origin
            break

    # Titre — classe style17
    titre_tag = soup.find('span', class_="style17")
    if titre_tag:
        infos["titre"] = titre_tag.get_text(strip=True)

    # Auteur : trouver le <td> contenant « dacteur » puis lire le <td> suivant
    for td in soup.find_all('td'):
        span = td.find('span', class_="style28")
        if span and "dacteur" in span.get_text():
            next_td = td.find_next_sibling('td')
            if next_td:
                match = re.search(r'-\s*(.+?)-\s*email', next_td.get_text())
                if match:
                    infos["auteur"] = match.group(1).strip()
                    break

    # Contact
    for td in soup.find_all('td'):
        span = td.find('span', class_="style28")
        if span and "contact" in span.get_text():
            next_td = td.find_next_sibling('td')
            if next_td:
                infos["contact"] = next_td.get_text(strip=True)

    # Images : source et légende
    for img_tag in soup.find_all("img", style=lambda s: s and "margin-bottom:5px" in s):
        img_src = img_tag.get('src')
        parent = img_tag.parent
        find_node = parent if (parent and parent.name == "a") else img_tag
        br_legend = find_node.find_next_sibling('br')
        if br_legend:
            span_legend = br_legend.find('span', class_="style21")
            if span_legend:
                infos["images"].append((img_src, span_legend.get_text(strip=True)))

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
    # Le titre se trouve dans un <span class="style17">
    titre = tree.xpath("//span[@class='style17']/text()")
    return titre[0]


def num_article_parse(file_content):
    """Extrait le numéro d'article à partir de l'URL présente dans le HTML.

    Args:
        file_content (str): Contenu HTML brut du fichier article.

    Returns:
        str | None: Numéro d'article extrait, ou ``None`` si non trouvé.
    """
    parser = etree.HTMLParser()
    tree = etree.fromstring(file_content, parser)
    # Le numéro d'article se trouve dans un <span class="style88">
    num_article_raw = tree.xpath("//span[@class='style88']/text()")
    match = re.search(r'/(\d+)\.htm$', num_article_raw[0])
    if match:
        return match.group(1)


def texte_parse(file_content):
    """Extrait et nettoie le texte de l'article via XPath.

    Args:
        file_content (str): Contenu HTML brut du fichier `.htm`.

    Returns:
        str: Texte de l'article concaténé sans sauts de ligne parasites.
    """
    parser = etree.HTMLParser()
    tree = etree.fromstring(file_content, parser)
    texte = tree.xpath("//p[@class='style96']//span[@class='style95']/text()")
    # Supprimer les sauts de ligne et concaténer les paragraphes
    return "".join([para for para in texte if para != "\n"])


def Creer_Corpus(archive_path):
    """Construit un corpus XML à partir des fichiers HTML d'un dossier.

    Parcourt tous les `.htm`, extrait les informations avec BeautifulSoup et
    XPath (en raison du travail en binôme, différentes méthodes ont été
    utilisées), puis écrit le résultat agrégé dans `corpus.xml`.

    Args:
        archive_path (str): Chemin du dossier contenant les fichiers `.htm`.
    """
    corpus_root = ET.Element("corpus")

    for file_path in glob.glob(archive_path + "/*.htm"):
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        doc = ET.SubElement(corpus_root, "document")
        infos = extraire_infos(html_content)

        ET.SubElement(doc, "article").text = num_article_parse(html_content)
        ET.SubElement(doc, "bulletin").text = infos["bulletin"]
        ET.SubElement(doc, "date").text = infos["date"]
        ET.SubElement(doc, "rubrique").text = infos["rubrique"]
        ET.SubElement(doc, "titre").text = titre_parse(html_content)
        ET.SubElement(doc, "auteur").text = infos["auteur"]
        ET.SubElement(doc, "texte").text = texte_parse(html_content)

        images_element = ET.SubElement(doc, "images")
        for img_url, legend in infos["images"]:
            ET.SubElement(images_element, "image").text = img_url
            ET.SubElement(images_element, "legend").text = legend

        ET.SubElement(doc, "contact").text = infos["contact"]

    xml_str = ET.tostring(corpus_root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    xml_indent = dom.toprettyxml(indent="    ")
    with open(TD1_DIR / "corpus.xml", "w", encoding="utf-8") as f:
        f.write(xml_indent)


if __name__ == "__main__":
    Creer_Corpus(get_path())
