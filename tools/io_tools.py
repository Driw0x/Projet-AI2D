import pandas as pd
import numpy as np
import json

# Lecture des donnees

def read_data(path):
    """
    Lit un fichier de donnees selon son extension.
    Formats geres :
    - json
    - csv
    """
    format = path.split(".")[-1]

    match format:
        case "json":
            df = pd.read_json(path)
        case "csv":
            df = pd.read_csv(path)
        case _:
            print("Format incorrect")
            return

    return df

# Preparation des donnees AlgoPython

def explode_json_column(df, column, max_level=None):
    """
    Éclate une colonne contenant une liste de dictionnaires JSON,
    puis transforme les clés des dictionnaires en colonnes pandas.
    """
    df = df.explode(column, ignore_index=True)

    details = pd.json_normalize(df[column], max_level=max_level)

    df = df.drop(columns=[column])
    df = df.join(details)

    return df


def trajectories_to_long(comptes):
    """
    Transforme la colonne 'trajectories' en format long.

    Avant :
        une ligne = un étudiant
        trajectories = {
            "A1": [tentative1, tentative2],
            "B7": [tentative1, tentative2]
        }

    Après :
        une ligne = un couple étudiant / exercice
        colonnes :
        - infos étudiant
        - level_1 = exercice
        - tentatives = liste des tentatives
    """
    lignes = []

    for _, row in comptes.iterrows():
        trajectories = row["trajectories"]

        # Certains comptes peuvent ne pas avoir de trajectoires valides.
        if not isinstance(trajectories, dict):
            continue

        # On garde toutes les colonnes de contexte :
        # id_compte, classe, etc.
        contexte = row.drop(labels=["trajectories"]).to_dict()

        # Chaque clé du dictionnaire est un exercice.
        # Chaque valeur est la liste des tentatives de cet exercice.
        for exercice, tentatives in trajectories.items():
            lignes.append({
                **contexte,
                "level_1": exercice,
                "tentatives": tentatives
            })

    return pd.DataFrame(lignes)


def AlgoPython_data(df):
    """
    Transforme les données brutes AlgoPython en DataFrame plat.

    Le JSON d'origine est imbriqué sous cette forme :

        classes
            -> comptes / étudiants
                -> trajectories
                    -> exercices
                        -> tentatives

    La sortie contient une ligne par tentative d'un étudiant
    sur un exercice.
    """

    # Copie pour éviter de modifier le DataFrame d'origine.
    data = df.copy()

    # 1) Éclater les classes.
    # Si une ligne contient plusieurs classes, on obtient une ligne par classe.
    classes = explode_json_column(data, "classes")

    # 2) Éclater les comptes.
    # Chaque classe contient plusieurs comptes étudiants.
    # On obtient une ligne par compte étudiant.
    comptes = explode_json_column(classes, "comptes", max_level=0)

    # 3) Transformer les trajectoires en format long.
    # Chaque exercice d'un étudiant devient une ligne.
    traj_long = trajectories_to_long(comptes)

    # 4) Éclater les tentatives.
    # Avant : une ligne contient une liste de tentatives.
    # Après : une ligne correspond à une tentative.
    df_final = traj_long.explode("tentatives", ignore_index=True)

    # 5) Normaliser les tentatives.
    # Chaque tentative est un dictionnaire :
    # {
    #   "code": "...",
    #   "statut": "...",
    #   "temps_passe": ...
    # }
    #
    # json_normalize transforme ces clés en colonnes.
    tentatives = pd.json_normalize(
        df_final["tentatives"].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
    )

    # 6) Remplacer la colonne brute "tentatives"
    # par les colonnes normalisées : code, statut, temps_passe, etc.
    df_final = df_final.drop(columns=["tentatives"]).join(tentatives)

    # 7) Garder uniquement les lignes où un code existe.
    df_final = df_final[df_final["code"].notna()]

    # 8) Retirer les statuts non exploitables.
    # "err" : erreur système ou tentative inexploitable.
    # "ask" : demande d'aide, pas une vraie soumission de code.
    df_final = df_final[~df_final["statut"].isin(["err", "ask"])]

    # 9) Transformer le statut en binaire.
    # ok      -> 1
    # autre   -> 0
    df_final["statut"] = np.where(df_final["statut"] == "ok", 1, 0)

    # 10) Nettoyer l'index.
    df_final = df_final.reset_index(drop=True)

    return df_final

# Sauvegarde

def clean_value(v):
    """
    Nettoie une valeur pour la rendre JSON compatible.
    """

    # numpy scalaires (à faire AVANT pd.isna)
    if isinstance(v, np.integer):
        return int(v)

    if isinstance(v, np.floating):
        if np.isnan(v):
            return None
        return float(v)

    if isinstance(v, np.bool_):
        return bool(v)

    # types python simples
    if isinstance(v, (int, float, str, bool)):
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return v

    # set -> liste
    if isinstance(v, set):
        return [clean_value(x) for x in v]

    # listes / tuples
    if isinstance(v, (list, tuple)):
        return [clean_value(x) for x in v]

    # dict
    if isinstance(v, dict):
        return {k: clean_value(val) for k, val in v.items()}

    # fallback
    return str(v)

def save_dataset_to_json(df, path):
    """
    Sauvegarde un dataset (DataFrame) en JSON.
    """

    # conversion DataFrame -> liste de dict
    records = df.to_dict(orient="records")

    # nettoyage JSON safe
    clean_records = [clean_value(row) for row in records]

    # sauvegarde
    with open("data/" + path, "w", encoding="utf-8") as f:
        json.dump(clean_records, f, indent=2, ensure_ascii=False)

    print(f"Dataset sauvegardé dans {path}")