# LO17 — Indexation et recherche d'information

Construction d'un moteur de recherche complet pour le corpus des Bulletins
électroniques, du scraping HTML au moteur de requêtes en langage naturel. 

Le projet est découpé en six parties successifs, chacun construit sur le
précédent : extraction du corpus, anti-dictionnaire, lemmatisation et index
inverse, correction orthographique, parsing de requête, moteur de recherche
et évaluation.

## Package requis

- `beautifulsoup4`, `lxml`, `spacy` (avec le modèle `fr_core_news_sm`),
  `nltk`, `matplotlib`
- Modèle spaCy : `python -m spacy download fr_core_news_sm`

## TD1 — Construction du corpus XML

Extraction des métadonnées (bulletin, date, rubrique, titre, auteur, texte,
images, contact) depuis les fichiers HTML des bulletins.

`TD1/TD1.py` | Parcourt `TD1/BULLETINS/`, parse chaque `.htm` et écrit `TD1/corpus.xml`


**Résultat d'exécution :** `TD1/corpus.xml` contenant tous les articles structurés en éléments `<document>`.

**Méthode d’exécution** : exécuter directement le fichier main.

## TD2 — Anti-dictionnaire par TF-IDF brut

Calcul de l'anti-dictionnaire sur les tokens bruts du corpus.

| Fichier | Rôle |
|---|---|
| `TD2/TD2.py` | Tokenise (spaCy), calcule TF-IDF, identifie les mots sous le seuil et produit le corpus filtré. |

**Résultats :** `TD2/anti_dict.txt` (mots vides et hors-domaine),
`TD2/corpus_filtre.xml` (corpus nettoyé), `TD2/tf-idf.txt` (scores).

## TD3 — Lemmatisation et index inverse

Trois scripts indépendants. Les sorties servent ensuite à TD5 et TD6.

| Fichier | Rôle |
|---|---|
| `TD3/1_1.py` | Compare la racinisation Snowball à la lemmatisation spaCy ; affiche les taux de compression (lemmes uniques / formes). |
| `TD3/2_1.py` | Anti-dictionnaire après lemmatisation spaCy + TF-IDF, plus précis que TD2. |
| `TD3/3_1.py` | Construit l'index inverse multi-zones (`titre`, `texte`, `auteur`, `date`, `rubrique`, `article`, `has_image`). |

**Résultats :** `TD3/anti_dict.txt`, `TD3/mot_lemma_list.txt`,
`TD3/tf-idf.txt`, `TD3/inverse_index.txt` (lemmes seuls),
`TD3/inverse_index_full.txt` (toutes zones), `TD3/corpus_filtre.xml`.

## TD4 — Correction orthographique

| Fichier | Rôle |
|---|---|
| `TD4/query_analyzer.py` | Tokenise une requête, valide chaque token contre le lexique, propose des corrections par préfixe commun et distance de Levenshtein pour les tokens hors-lexique. |

**Résultat :** affichage console des suggestions de correction par token. Le lexique de référence est `TD4/mini_lexicon.tsv`.

## TD5 — Pipeline de traitement de requête

| Fichier | Rôle |
|---|---|
| `TD5/traitement_requete.py` | Analyse une requête en langage naturel et produit la métadonnée structurée consommée par TD6 : groupes DNF labellisés `{title, content}`, exclusions, filtres de rubrique, de date, d'image. |

Pickles mis en cache : `TD5/lemma_dict.pkl`, `TD5/anti_list.pkl`,
`TD5/tf-idf.pkl`. Requêtes de test : `TD5/requete.txt`.

**Résultat d'exécution :** la métadonnée structurée de chaque requête est
imprimée. Exemple :

```
Je voudrais les articles qui parlent d'airbus ou du projet Taxibot
key_word_groups: [{'title': [], 'content': ['airbus']},
                  {'title': [], 'content': ['projet', 'taxibot']}]
```

## TD6 — Moteur de recherche et évaluation

| Fichier | Rôle |
|---|---|
| `TD6/moteur.py` | Charge l'index et le corpus, évalue une requête via le pipeline TD5, score les bulletins, résout les articles, affiche les extraits avec mots-clés mis en évidence. Mode CLI interactif. |


**Résultats d'exécution :**


Graphiques : `TD6/eval_precision_recall.png`, `TD6/eval_timing.png`.

Le mode CLI interactif (`python3 TD6/moteur.py`) accepte une requête en
langage naturel et liste les articles correspondants avec leurs extraits.

