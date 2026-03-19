from bs4 import BeautifulSoup
import re
import os
import glob
import sys

def extraire_infos(filepath) :
    infos = {
      "article" : "",
      "bulletin" : "",
      "date" : "",
      "rubrique" : "",
      "titre" : "",
      "auteur" : "",
      "texte" : "",
      "images_url" : [],
      "images_legend" : [], 
      "contact" : "",
    }

    with open(filepath, 'r', encoding = "utf-8") as f :
        contenue = f.read()
    soup = BeautifulSoup(contenue, "html.parser")
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
        match_date = re.search(r'(\d{2}/\d{2}/\d{4})', text_origin)
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
    ## first find td with span stye 28 and with "/s+dacteur", then find the sibling span
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
        infos["images_url"].append(img_src)
        parent = img_tag.parent
        if parent and parent.name == "a" :
            find_node = parent
        else:
            find_node = img_tag
        br_legend = find_node.find_next_sibling('br')
        if br_legend : 
            span_legend = br_legend.find('span', class_ = "style21")
            if span_legend :
                infos["images_legend"].append(span_legend.get_text(strip = True))

    return infos
        



if __name__ == '__main__':
    dossier = "TD1/BULLETINS"
    fichiers = glob.glob(os.path.join(dossier, "*.htm"))
    for fichier in fichiers :
        print(f" Traitement de : {fichier}")
        infos = extraire_infos(fichier)
        print(infos)

