

def pipeline_traitement_requete(source : str, metadonnees : dict) -> dict:
    metadonnees = traiter_metadonnees(source, metadonnees)
    metadonnees = traiter_op_logique(source, metadonnees)
    metadonnees = traiter_mots_cles(source, metadonnees)
    return metadonnees





def traiter_metadonnees(source : str, metadonnees : dict) -> dict:
    '''
    Traitez les métadonnées ici, incluant les Les contraintes temporelles, Les rubriques spécifiques de l’ADIT,
    
    Les filtres structurels
    Args:
        source : la requete a traiter
        metadonnees : dictionnaire contenant les métadonnées extraites de la requete

    Returns:
        dict: un dictionnaire contenant les métadonnées traitées
    '''
    return metadonnees  # Retournez les métadonnées traitées



def traiter_op_logique(source : str, metadonnees : dict) -> dict:
    '''
    Traitez les opérateurs logiques ici, incluant les opérateurs logiques (AND, OR, NOT)

    Args:
        source : la requete a traiter
        metadonnees : dictionnaire contenant les métadonnées extraites de la requete

    Returns:
        dict: un dictionnaire contenant les métadonnées traitées
    '''


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

