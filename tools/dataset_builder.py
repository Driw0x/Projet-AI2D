import numpy as np
from tools.ast_tools import code_to_ast
import pandas as pd
from tqdm import tqdm
from tools.io_tools import save_dataset_to_json
from tools.comparaison import compare_transition

def check_required_columns(df, columns):
    """
    Vérifie que les colonnes nécessaires sont présentes dans un DataFrame.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes : {missing}. Colonnes disponibles : {df.columns.tolist()}")


def prepare_attempts_dataframe(dfo, min_tentatives=5):
    """
    Filtre les tentatives exploitables :
    - code présent et non vide ;
    - couple (user, exercice) avec assez de tentatives.
    """
    check_required_columns(dfo, ["id_compte", "level_1", "code"])

    df = dfo[dfo["code"].notna() & (dfo["code"] != "")].copy()
    valid_couples = couples_valides(df, min_tentatives=min_tentatives)

    return df[df[["id_compte", "level_1"]].apply(tuple, axis=1).isin(valid_couples)].copy()


def build_ast_solutions(sol_info):
    """
    Convertit les codes solutions d'un exercice en AST.
    """
    if sol_info is None:
        return []

    ast_solutions = []
    for sol in sol_info.get("correctCodes", []):
        ast_sol = code_to_ast(sol)
        if ast_sol is not None:
            ast_solutions.append(ast_sol)

    return ast_solutions


def safe_temps(value):
    """
    Normalise un temps invalide ou nul en NaN.
    """
    if pd.isna(value) or value == 0:
        return np.nan
    return value


def build_solution_dict(solution_data):
    """
    Construit un dictionnaire des solutions par exercice.

    Retour:
    dict{exerciseTitle: {"correctCodes","exerciseType"}}
    """
    solution_dict = {}

    for _, row in solution_data.iterrows():
        solution_dict[row["exerciseTitle"].split(" ")[0].replace("-", "")] = {
            "correctCodes": row["correctCodes"],
            "exerciseType": row["exerciseType"]
        }

    return solution_dict

def build_user_exercise_count_dict(df):
    """
    Calcule le nombre de tentatives non vides pour chaque couple (user, exercice).

    Retour :
    - dict {(id_compte, exercice): nb_tentatives}
    """
    required = {"id_compte", "level_1", "code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes absentes : {sorted(missing)}")

    data = df[df["code"].notna() & (df["code"] != "")].copy()

    return data.groupby(["id_compte", "level_1"], dropna=False).size().to_dict()


def couples_valides(df, min_tentatives=5):
    """
    Retourne les couples (id_compte, exercice) ayant plus de min_tentatives tentatives.
    Par défaut : > 5 comme dans l'ancien code.
    """
    counts = build_user_exercise_count_dict(df)
    return {key for key, nb in counts.items() if nb > min_tentatives}


def build_transition_dataset(
    dfo,
    solution_df,
    include_code_errors=False,
    include_solution_scores=True,
    include_temps=True,
    include_status=True,
    include_codes=False,
    min_tentatives=5,
    save_path=None,
):
    """
    Construit le dataset commun des transitions t -> t+1.
    """
    required = ["id_compte", "level_1", "code"]
    if include_temps:
        required.append("temps_passe")
    if include_status:
        required.append("statut")
    check_required_columns(dfo, required)

    df = prepare_attempts_dataframe(dfo, min_tentatives=min_tentatives)
    solution_dict = build_solution_dict(solution_df)
    lignes = []

    user_groups = list(df.groupby("id_compte", sort=False))

    for user_id, user_df in tqdm(user_groups, desc="Processing", position=0, leave=True, dynamic_ncols=True):
        ex_groups = list(user_df.groupby("level_1", sort=False))

        for ex, group in tqdm(ex_groups, desc=f"User {user_id}", position=1, leave=False):
            group = group.reset_index(drop=True)
            if len(group) < 2:
                continue

            sol_info = solution_dict.get(ex)
            exercise_type = sol_info.get("exerciseType", np.nan) if sol_info is not None else np.nan
            ast_solutions = build_ast_solutions(sol_info) if include_solution_scores else []
            zss_cache = {}
            reussite_finale_exercice = group.loc[len(group) - 1, "statut"] if include_status else np.nan

            for i in tqdm(range(len(group) - 1), desc=f"Exercice {ex}", position=2, leave=False):
                code_t = group.loc[i, "code"]
                code_t_1 = group.loc[i + 1, "code"]
                context = f"user={user_id}, ex={ex}, t={i + 1}"

                comparison = compare_transition(
                    code_t,
                    code_t_1,
                    zss_cache=zss_cache,
                    ast_solutions=ast_solutions,
                    include_code_errors=include_code_errors,
                    include_solution_scores=include_solution_scores,
                    context=context,
                )

                row = {
                    "id": user_id,
                    "exercice": ex,
                    "type_exercice": exercise_type,
                    "t": i + 1,
                    "t_plus_1": i + 2,
                    "distance_zss": comparison["distance_zss"],
                    "ops": comparison["ops"],
                }

                if include_codes:
                    row["code_t"] = code_t
                    row["code_t_1"] = code_t_1

                if include_code_errors:
                    row.update({
                        "primary_code_errors_score": comparison["primary_code_errors_score"],
                        "primary_code_errors": comparison["primary_code_errors"],
                        "typology_based_code_error_score": comparison["typology_based_code_error_score"],
                        "typology_based_code_error": comparison["typology_based_code_error"],
                    })

                if include_solution_scores:
                    row.update({
                        "score_t_solution": comparison["score_t_solution"],
                        "score_t_plus_1_solution": comparison["score_t_plus_1_solution"],
                        "progression_solution": comparison["progression_solution"],
                    })

                if include_temps:
                    temps_t = safe_temps(group.loc[i, "temps_passe"])
                    temps_t_1 = safe_temps(group.loc[i + 1, "temps_passe"])
                    row["temps_t"] = temps_t
                    row["temps_t_plus_1"] = temps_t_1
                    row["delta_temps"] = temps_t_1 - temps_t if pd.notna(temps_t) and pd.notna(temps_t_1) else np.nan

                if include_status:
                    row.update({
                        "statut_t": group.loc[i, "statut"],
                        "statut_t_plus_1": group.loc[i + 1, "statut"],
                        "reussite_finale_exercice": reussite_finale_exercice,
                    })

                lignes.append(row)

    dataset = pd.DataFrame(lignes)

    if save_path is not None:
        save_dataset_to_json(dataset, save_path)

    return dataset