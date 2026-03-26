# TD4 - Analyseur de requete avec correction

## Fichiers

- `query_analyzer.py` : script principal.
- `mini_lexicon.tsv` : petit lexique manuel pour les premiers tests.

## Fonctionnement

Le programme realise les etapes demandees :

1. saisie d'une requete au clavier, ou via `--query` ;
2. tokenisation de la phrase complete ;
3. parcours des termes un a un ;
4. conservation directe des entites specifiques (nombres, dates) ;
5. validation directe si le terme existe dans le lexique ;
6. generation de candidats par recherche de prefixe avec trois hyperparametres :
   - `seuilMin`
   - `seuilMax`
   - `seuilProximite`
7. retour direct si un seul candidat est trouve ;
8. departage par distance de Levenshtein s'il y a plusieurs candidats ;
9. affichage d'un message explicite si aucun candidat n'est trouve.

## Commandes utiles

Petit lexique manuel :

```bash
python query_analyzer.py
python query_analyzer.py --query "innovatoin textile 2024 12/10/2025"
```

Lexique complet ADIT :

```bash
python query_analyzer.py --lexicon full
python query_analyzer.py --lexicon full --query "internationel textil octobre 2024"
```

## Hyperparametres par defaut

- `seuilMin = 2`
- `seuilMax = 3`
- `seuilProximite = 0.4`

## Remarque technique

Le script essaie d'utiliser `spaCy` et `fr_core_news_sm` si disponibles, pour rester coherent avec les TD precedents.
Si `spaCy` n'est pas installe, il bascule automatiquement sur une tokenisation integree afin que le programme reste executable sans dependance supplementaire.
