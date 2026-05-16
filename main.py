"""
Point d'entrée principal du projet AlgoPython.

Ce fichier sert à lancer le pipeline d'analyse :
1. charger les données brutes ;
2. préparer le DataFrame AlgoPython ;
3. générer ou recharger les datasets intermédiaires ;
4. lancer la classification ;
5. afficher les résultats et les visualisations.

Les fonctions de traitement sont dans le dossier `tools/`.
Ce fichier ne contient donc que l'orchestration du projet.
"""

import numpy as np
from pandasgui import show

from tools.io_tools import read_data, AlgoPython_data
from tools.ast_tools import code_to_ast, ast_dump
from tools.comparaison import primary_code_error_two_prog
from tools.cas import cas1, cas2
from tools.classification import pre_calcul, build_user_classification, build_user_exercice_classification, distribution_classes
import tools.visualisations as viz


# =========================
# Chemins des fichiers
# =========================

DATA_PATH = "data/2025.json"
EXERCISES_PATH = "data/exercises.json"

CAS1_PATH = "data/cas1.json"
CAS2_PATH = "data/cas2.json"
PRE_CALCUL_PATH = "data/pre_calcul.json"


# =========================
# Chargement / préparation
# =========================

def load_project_data():
    """
    Charge les données brutes puis construit le DataFrame plat AlgoPython.

    Retourne :
    - df : tentatives étudiantes préparées ;
    - solutions : solutions attendues des exercices.
    """
    raw_data = read_data(DATA_PATH)
    solutions = read_data(EXERCISES_PATH)

    df = AlgoPython_data(raw_data)

    return df, solutions


# =========================
# Fonctions d'inspection
# =========================

def display_ast_for_row(df, idx=0):
    """
    Affiche l'AST du code présent à la ligne `idx`.

    Utile pour comprendre comment Python représente un programme
    avant les calculs de comparaison ou de distance ZSS.
    """
    code = df.iloc[idx]["code"]
    tree = code_to_ast(code)

    print(ast_dump(tree))


def display_program_comparison(df, idx1=0, idx2=1):
    """
    Compare deux codes du DataFrame avec l'outil AED
    et affiche les erreurs principales détectées.
    """
    code1 = df.iloc[idx1]["code"]
    code2 = df.iloc[idx2]["code"]

    result = primary_code_error_two_prog(code1, code2)

    print(result)


def show_non_empty_codes(df):
    """
    Ouvre dans PandasGUI uniquement les lignes qui contiennent du code.
    """
    show(df[df["code"] != ""])


# =========================
# Datasets intermédiaires
# =========================

def get_cas1_dataset(df, solutions, reload=False):
    """
    Retourne le dataset cas1.

    Si reload=True, recharge le fichier déjà calculé.
    Sinon, recalcule cas1 à partir des données préparées.
    """
    if reload:
        return read_data(CAS1_PATH)

    return cas1(df, solutions)


def get_cas2_dataset(cas1_df, reload=False):
    """
    Retourne le dataset cas2.

    Si reload=True, recharge le fichier déjà calculé.
    Sinon, enrichit cas1 avec les commentaires textuels.
    """
    if reload:
        return read_data(CAS2_PATH)

    return cas2(cas1_df)


def get_pre_calcul_dataset(df, solutions, reload=True):
    """
    Retourne le dataset de pré-calcul utilisé pour la classification.

    Si reload=True, recharge le fichier déjà calculé.
    Sinon, recalcule les transitions, distances, scores et temps.
    """
    if reload:
        return read_data(PRE_CALCUL_PATH)

    return pre_calcul(df, solutions)


# =========================
# Analyse détaillée
# =========================

def display_user_exercise_analysis(cas2_df, user_id=None, exercice=None):
    """
    Affiche les transitions détaillées d'un utilisateur sur un exercice.

    Si aucun utilisateur ou exercice n'est donné, un couple existant
    est choisi aléatoirement.
    """
    if user_id is None:
        user_id = np.random.choice(cas2_df["id"], 1)[0]

    if exercice is None:
        exercice = np.random.choice(
            cas2_df[cas2_df["id"] == user_id]["exercice"],
            1,
        )[0]

    user_exercise_df = cas2_df[
        (cas2_df["id"] == user_id)
        & (cas2_df["exercice"] == exercice)
    ].reset_index(drop=True)

    print(f"\n===== UTILISATEUR {user_id} / EXERCICE {exercice} =====\n")

    for i in range(len(user_exercise_df)):
        print(f"\n--- Transition tentative {i + 1} -> {i + 2} ---\n")

        print("Code t :\n")
        print(user_exercise_df.loc[i, "code_t"])

        print("\nCode t+1 :\n")
        print(user_exercise_df.loc[i, "code_t_1"])

        primary_text = user_exercise_df.loc[i, "primary_code_errors_text"]
        typology_text = user_exercise_df.loc[i, "typology_based_code_error_text"]

        if len(typology_text) > 0:
            print("\nModification simple :")
            for txt in typology_text:
                print("-", txt)

        if len(primary_text) > 0:
            print("\nModification détaillée :")
            for txt in primary_text:
                print("-", txt)

        if len(primary_text) == 0 and len(typology_text) == 0:
            print("\nPas de modification détectée")

        print("\n==============================")


# =========================
# Classification
# =========================

def run_classification_analysis(pre_calcul_df, display_tables=True, display_plots=True):
    """
    Lance les classifications :
    - globale par utilisateur ;
    - détaillée par couple utilisateur/exercice.

    Affiche aussi les seuils et les distributions de classes.
    """
    user_stats, user_thresholds = build_user_classification(pre_calcul_df)

    user_exercise_stats, user_exercise_thresholds = (
        build_user_exercice_classification(pre_calcul_df)
    )

    print("\n===== Seuils classification utilisateur =====")
    print(user_thresholds)

    print("\n===== Seuils classification utilisateur/exercice =====")
    print(user_exercise_thresholds)

    print("\n===== Distribution des classes utilisateur =====")
    print(distribution_classes(user_stats))

    print("\n===== Distribution des classes utilisateur/exercice =====")
    print(distribution_classes(user_exercise_stats))

    if display_tables:
        show(user_stats)
        show(user_exercise_stats)

    if display_plots:
        viz.run_all_plots(pre_calcul_df)

    return user_stats, user_exercise_stats


# =========================
# Pipeline principal
# =========================

def main():
    """
    Lance le pipeline principal du projet.

    Par défaut :
    - charge les données ;
    - recharge le pré-calcul existant ;
    - lance la classification ;
    - affiche les tableaux et graphiques.

    Les blocs commentés peuvent être activés selon l'analyse voulue.
    """
    df, solutions = load_project_data()

    # Inspection AST d'une tentative.
    # display_ast_for_row(df, idx=0)

    # Comparaison directe entre deux codes.
    # display_program_comparison(df, idx1=0, idx2=1)

    # Affichage des codes non vides.
    # show_non_empty_codes(df)

    # Génération ou chargement de cas1.
    # cas1_df = get_cas1_dataset(df, solutions, reload=False)
    # show(cas1_df)

    # Génération ou chargement de cas2.
    # cas2_df = get_cas2_dataset(cas1_df, reload=False)
    # show(cas2_df)

    # Analyse détaillée d'un utilisateur sur un exercice.
    # display_user_exercise_analysis(
    #     cas2_df,
    #     user_id=22933,
    #     exercice="B7",
    # )

    # Pré-calcul pour la classification.
    pre_calcul_df = get_pre_calcul_dataset(
        df,
        solutions,
        reload=True,
    )

    # Classification et visualisations.
    run_classification_analysis(
        pre_calcul_df,
        display_tables=True,
        display_plots=True,
    )


if __name__ == "__main__":
    main()
