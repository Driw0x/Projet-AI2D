import utils
import numpy as np
from pandasgui import show


# =========================
# Configuration
# =========================

DATA_PATH = "data/2025.json"
EXERCISES_PATH = "data/exercises.json"

CAS1_PATH = "data/cas1.json"
CAS2_PATH = "data/cas2.json"
PRE_CALCUL_PATH = "data/pre_calcul.json"


# =========================
# Chargement des données
# =========================

def load_data():
    """
    Charge et prépare les données AlgoPython.
    """
    data = utils.read_data(DATA_PATH)
    solutions = utils.read_data(EXERCISES_PATH)

    df = utils.AlgoPython_data(data)

    return df, solutions


# =========================
# Outils debug / AST
# =========================

def debug_ast(df, idx=0):
    """
    Affiche l'AST d'un programme.
    """
    code = df.iloc[idx]["code"]

    ast_tree = utils.code_to_ast(code)

    print(utils.ast_dump(ast_tree))


def compare_programs(df, idx1=0, idx2=1):
    """
    Compare deux programmes.
    """
    code1 = df.iloc[idx1]["code"]
    code2 = df.iloc[idx2]["code"]

    result = utils.primary_code_error_two_prog(code1, code2)

    print(result)


# =========================
# Affichage dataset
# =========================

def show_codes(df):
    """
    Affiche uniquement les lignes avec du code.
    """
    show(df[df["code"] != ""])


# =========================
# Cas 1
# =========================

def build_cas1(df, solutions, reload=False):
    """
    Génère ou recharge cas1.
    """
    if reload:
        return utils.read_data(CAS1_PATH)

    return utils.cas1(df, solutions)


# =========================
# Cas 2
# =========================

def build_cas2(cas1_df, reload=False):
    """
    Génère ou recharge cas2.
    """
    if reload:
        return utils.read_data(CAS2_PATH)

    return utils.cas2(cas1_df)


# =========================
# Analyse d'un user/exercice
# =========================

def analyse_user_exercice(cas2_df, user_id=None, exercice=None):
    """
    Affiche les transitions d'un utilisateur.
    """

    if user_id is None:
        user_id = np.random.choice(cas2_df["id"], 1)[0]

    if exercice is None:
        exercice = np.random.choice(
            cas2_df[cas2_df["id"] == user_id]["exercice"],
            1
        )[0]

    cas = cas2_df[
        (cas2_df["id"] == user_id)
        & (cas2_df["exercice"] == exercice)
    ].reset_index(drop=True)

    print(f"\n===== USER {user_id} / EXERCICE {exercice} =====\n")

    for i in range(len(cas)):

        print(f"\n--- Tentative {i+1} -> {i+2} ---\n")

        print("Code t :\n")
        print(cas.loc[i, "code_t"])

        print("\nCode t+1 :\n")
        print(cas.loc[i, "code_t_1"])

        primary_text = cas.loc[i, "primary_code_errors_text"]
        typology_text = cas.loc[i, "typology_based_code_error_text"]

        if len(typology_text) > 0:
            print("\nModification simple :")
            for txt in typology_text:
                print("-", txt)

        if len(primary_text) > 0:
            print("\nModification détaillée :")
            for txt in primary_text:
                print("-", txt)

        if len(primary_text) == 0 and len(typology_text) == 0:
            print("\nPas de modification")

        print("\n==============================")


# =========================
# Pré-calcul
# =========================

def build_pre_calcul(df, solutions, reload=True):
    """
    Génère ou recharge le pré-calcul.
    """
    if reload:
        return utils.read_data(PRE_CALCUL_PATH)

    return utils.pre_calcul(df, solutions)


# =========================
# Classification
# =========================

def test_classification(pre_calcul_df):
    """
    Lance les classifications.
    """

    user_stats, seuils_user = utils.build_user_classification(pre_calcul_df)

    user_ex_stats, seuils_user_ex = (
        utils.build_user_exercice_classification(pre_calcul_df)
    )

    print("\n===== Seuils user =====")
    print(seuils_user)

    print("\n===== Seuils user/exercice =====")
    print(seuils_user_ex)

    print("\n===== Distribution user =====")
    print(utils.distribution_classes(user_stats))

    print("\n===== Distribution user/exercice =====")
    print(utils.distribution_classes(user_ex_stats))

    show(user_stats)
    show(user_ex_stats)


# =========================
# Main
# =========================

def main():

    df, solutions = load_data()

    # =====================
    # Debug AST
    # =====================

    # debug_ast(df, idx=0)

    # =====================
    # Comparaison programmes
    # =====================

    # compare_programs(df, idx1=0, idx2=1)

    # =====================
    # Affichage dataset
    # =====================

    # show_codes(df)

    # =====================
    # Cas1
    # =====================

    # cas1_df = build_cas1(df, solutions, reload=False)
    # show(cas1_df)

    # =====================
    # Cas2
    # =====================

    # cas2_df = build_cas2(cas1_df, reload=False)
    # show(cas2_df)

    # =====================
    # Analyse détaillée
    # =====================

    # analyse_user_exercice(
    #     cas2_df,
    #     user_id=22933,
    #     exercice="B7"
    # )

    # analyse_user_exercice(cas2_df)

    # =====================
    # Pré-calcul
    # =====================

    pre_calcul_df = build_pre_calcul(
        df,
        solutions,
        reload=True
    )

    # =====================
    # Classification
    # =====================

    test_classification(pre_calcul_df)


if __name__ == "__main__":
    main()