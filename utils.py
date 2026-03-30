import pandas as pd
import ast_error_detection as aed
import ast
import numpy as np
import json
import textwrap
from tqdm import tqdm
from ast_error_detection.zang_shasha_distance import distance


# Outils AST

def code_to_ast(code):
    """
    Transforme un code Python sous forme de texte en AST.
    AST = arbre syntaxique abstrait.
    """
    try:
        return ast.parse(code)

    except Exception:
        try:
            # normaliser indentation
            code_fixed = textwrap.dedent(code)
            return ast.parse(code_fixed)

        except Exception:
            return None

def ast_dump(t):
    """
    Affiche l'AST de manière lisible.
    Utile pour voir la structure du code.
    """
    return ast.dump(t, indent=2)

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

def AlgoPython_data(df):
    """
    Transforme les donnees brutes AlgoPython en DataFrame plat.

    etapes :
    - eclate la colonne classes
    - eclate la colonne comptes
    - recupère les trajectoires
    - eclate les tentatives
    - transforme le statut en binaire :
        ok -> 1
        autres -> 0
    """
    # copie du DataFrame
    classes = df[df.columns]

    # une ligne par classe
    classes = classes.explode("classes", ignore_index=True)

    # normalise le contenu JSON de la colonne classes
    classes_details = pd.json_normalize(classes["classes"])
    classes = classes.drop(columns=["classes"])
    classes = classes.join(classes_details)

    # une ligne par compte
    comptes = classes.explode("comptes", ignore_index=True)

    # normalise le contenu JSON de la colonne comptes
    comptes_details = pd.json_normalize(comptes["comptes"], max_level=0)
    comptes = comptes.drop(columns=["comptes"])
    comptes = comptes.join(comptes_details)

    # recupère les trajectoires
    trajectories = comptes["trajectories"]

    # garde seulement les trajectoires qui sont des dictionnaires
    trajectories = trajectories.where(trajectories.map(type).eq(dict), {})

    # passe les trajectoires en format large
    traj_wide = trajectories.apply(pd.Series)

    # passe en format long
    traj_long = traj_wide.stack(future_stack=True).reset_index(name="tentatives")

    # recupère les colonnes de contexte
    contexte = comptes.drop(columns=["trajectories"])
    traj_long = traj_long.join(contexte, on="level_0")
    traj_long = traj_long.drop(columns=["level_0"])

    # une ligne par tentative
    traj_long = traj_long.explode("tentatives", ignore_index=True)

    # normalise les tentatives
    tentatives = pd.json_normalize(traj_long["tentatives"].apply(lambda x: x if isinstance(x, dict) else {}))

    # joint les infos des tentatives au contexte
    df_final = traj_long.drop(columns=["tentatives"]).join(tentatives)

    # enlève les lignes sans code
    df_final = df_final[df_final["code"].notna()]

    # remet un index propre
    df_final = df_final.reset_index(drop=True)

    # enlève les statuts "err" et "ask"
    df_final = df_final[(df_final["statut"] != "err") & (df_final["statut"] != "ask")]

    # transforme ok en 1 et le reste en 0
    df_final["statut"] = np.where(df_final["statut"] == "ok", 1, 0)

    return df_final

# Comparaison de programmes

def primary_code_error_two_prog(p1, p2):
    """
    Compare deux programmes et retourne les erreurs principales detectees.
    En cas d'erreur dans l'outil, retourne [0, []].
    """
    try:
        return aed.get_primary_code_errors(p1, p2)
    except:
        return [0, []]


def prog_vs_answer(p1, list_answer):
    """
    Compare un programme à une ou plusieurs reponses attendues.
    Retourne la typologie d'erreurs.
    En cas d'erreur, retourne [0, {}].
    """
    try:
        return aed.get_typology_based_code_error(p1, list_answer)
    except:
        return [0, {}]

# Pre calcul

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
    Calcule le nombre de tentatives pour chaque couple (user, exercice).

    Retour :
    - dict {(id_compte, exercice): nb_tentatives}
    """
    df = df[df["code"] != ""]

    counts = {}

    for user_id in np.unique(df["id_compte"]):
        user_df = df[df["id_compte"] == user_id]

        for ex in np.unique(user_df["level_1"]):
            key = (
                int(user_id) if isinstance(user_id, np.generic) else user_id,
                str(ex)
            )
            counts[key] = len(user_df[user_df["level_1"] == ex])

    return counts

def couples_valides(df):
    all_ex = np.unique(df["level_1"])
    all_users = np.unique(df["id_compte"])
    couples = set()
    user_ex_counts = build_user_exercise_count_dict(df)

    for user_id in all_users:
        for ex in all_ex:
            key = (user_id, ex)
            nb_tentatives = user_ex_counts.get(key, 0)
            if nb_tentatives > 5:
                couples.add(key)

    return couples

# Wrapping

class Wrapper:
    """
    Wrapper d'un noeud AST pour la distance de Zhang-Shasha.
    """

    def __init__(self, ast_node, path=None):
        self.label = type(ast_node).__name__
        self.children = []

        if path is None:
            self._path = [self.label]
        else:
            self._path = path

        child_index = 0
        for _, value in ast.iter_fields(ast_node):
            if isinstance(value, ast.AST):
                child_path = self._path + [f"{type(value).__name__}[{child_index}]"]
                self.children.append(Wrapper(value, child_path))
                child_index += 1

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        child_path = self._path + [f"{type(item).__name__}[{child_index}]"]
                        self.children.append(Wrapper(item, child_path))
                        child_index += 1

    def get_path(self):
        return self._path

def code_to_zss_node(code):
    """
    Transforme un code Python en arbre compatible avec Zhang-Shasha.
    """
    return Wrapper(code_to_ast(code))

def get_children(node):
    """
    Retourne les enfants d'un noeud compatible Zhang-Shasha.
    """
    return node.children

def get_zss_tree(code, zss_cache):
    if code not in zss_cache:
        try:
            zss_cache[code] = code_to_zss_node(code)
        except Exception:
            zss_cache[code] = None
    return zss_cache[code]

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

# Cas 1: transformation t->t+1
def cas1_x_y(data, x, y, zss_cache=None):
    """
    Compare le code x et y.
    """
    if x < 0 or y < 0 or x >= len(data) or y >= len(data):
        return np.nan, {}, [0, {}]

    if zss_cache is None:
        zss_cache = {}

    code_x = data.loc[x, "code"]
    code_y = data.loc[y, "code"]

    # comparaison x -> y
    try:
        primary = primary_code_error_two_prog(code_x, code_y)
        typology = prog_vs_answer(code_x, [code_y])
    except Exception as err:
        tqdm.write(f"Erreur comparaison t={x+1}, t'={y+1}: {err}")
        primary = (0, [])
        typology = [0, {}]

    # distance ZSS
    tree_x = get_zss_tree(code_x, zss_cache)
    tree_y = get_zss_tree(code_y, zss_cache)

    if tree_x is None or tree_y is None:
        d, ops = np.nan, {}
    else:
        try:
            d, ops = distance(tree_x, tree_y, get_children)
        except Exception as err:
            tqdm.write(f"Erreur distance t={x+1}, t'={y+1}: {err}")
            d, ops = np.nan, {}

    return d, ops, primary, typology

def cas1(dfo, solution_df):
    """
    Construit le dataset des comparaisons t -> t+1.

    Retour:
        dataset_t_t1(
            id,
            exercice,
            type_exercice,
            t,
            dist_zss,
            ops,
            code_t,
            code_t_1,
            comparaison_t_t1
        )
    """
    df = dfo[dfo["code"].notna() & (dfo["code"] != "")].copy()

    couples = couples_valides(dfo)
    df = df[df[["id_compte", "level_1"]].apply(tuple, axis=1).isin(couples)].copy()

    solution_dict = build_solution_dict(solution_df)

    lignes = []

    user_groups = list(df.groupby("id_compte", sort=False))

    for user_id, user_df in tqdm(user_groups, desc="Processing", position=0, leave=True, dynamic_ncols=True):

        ex_groups = list(user_df.groupby("level_1", sort=False))

        for ex, group in tqdm(ex_groups,
                              desc=f"User {user_id}",
                              position=1,
                              leave=False):

            group = group.reset_index(drop=True)

            if len(group) < 2:
                continue

            sol_info = solution_dict.get(ex)
            exercise_type = sol_info.get("exerciseType", np.nan) if sol_info is not None else np.nan

            zss_cache = {}

            for i in tqdm(range(len(group) - 1),
                          desc=f"Exercice {ex}",
                          position=2,
                          leave=False):
                
                code_t = group.loc[i, "code"]
                code_t_1 = group.loc[i + 1, "code"]
                
                d, ops, primary, typology = cas1_x_y(group,
                                                     i,
                                                     i + 1,
                                                     zss_cache)
                
                row = {"id": user_id,
                       "exercice": ex,
                       "type_exercice": exercise_type,
                       "t": i + 1,
                       "dist_zss": d,
                       "ops": ops,
                       "code_t": code_t,
                       "code_t_1": code_t_1,
                       "primary_code_errors_score": primary[0],
                       "primary_code_errors": primary[1],
                       "typology_based_code_error_score": typology[0],
                       "typology_based_code_error": typology[1]}

                lignes.append(row)

    dataset = pd.DataFrame(lignes)

    save_dataset_to_json(dataset, "cas1.json")

    return dataset

# Cas 2
def regle(ops=None, primary_code_errors=None, typology_based_code_error=None, ):
    """
    Affiche une interpretation textuelle simple des erreurs detectees.
    """
    if ops != None:
        pass
    elif primary_code_errors != None:
        for errors in primary_code_errors:
            if errors[0].startswith("Missing"):
                print("Ajout")
            elif errors[0].startswith("UNNECESSARY"):
                print("Suppression")
            else:
                print("Mise à jour")
    elif typology_based_code_error != None:
        cas = set()
        for errors in typology_based_code_error:
            if errors.startswith("F_CALL_MISSING"):
                print(f"Ajout d'un appel à la fonction {errors.split('_')[-1]}")
            elif errors.startswith("F_CALL_UNNECESSARY"):
                print(f"Suppression d'un appel à la fonction {errors.split('_')[-1]}")
            else:
                """
                'F_CALL_INCORRECT_POSITION_AVANCER'
                'LO_BODY_ERROR', 'EXP_ERROR_ASSIGNMENT_MISSING'
                'F_CALL_PRINT_ERROR_ARG', 'F_CALL_COULEUR_ERROR'
                'F_CALL_INCORRECT_POSITION_PRINT', 'F_CALL_BAS_ERROR'
                'F_CALL_HAUT_ERROR', 'EXP_ERROR_OPERATOR', 'LO_FOR_NUMBER_ITERATION_ERROR'
                'F_DEFINITION_MISSING', 'EXP_ERROR_OPERANDS', 'F_CALL_ARC_ERROR'
                'F_CALL_INCORRECT_POSITION_ARC', 'F_CALL_INCORRECT_POSITION_TOURNER'
                'F_CALL_INCORRECT_POSITION_POSER', 'F_CALL_INCORRECT_POSITION_LEVER'
                'LO_FOR_MISPLACED', 'F_CALL_AVANCER_ERROR', 'F_CALL_GAUCHE_ERROR'
                'F_CALL_TOURNER_ERROR', 'LO_BODY_MISSING_NOT_PRESENT_ANYWHERE'
                'LO_FOR_UNNECESSARY', 'EXP_ERROR_ASSIGNMENT_UNNECESSARY'
                'F_CALL_DROITE_ERROR', 'LO_FOR_MISSING', 'F_CALL_INCORRECT_POSITION_COULEUR'
                'F_CALL_INCORRECT_POSITION_BAS', 'F_CALL_INCORRECT_POSITION_DROITE'
                'LO_BODY_MISPLACED', 'EXP_ERROR_OPERATION', 'F_CALL_INCORRECT_POSITION_GAUCHE'
                'CS_MISSING', 'LO_FOR_NUMBER_ITERATION_ERROR_UNDER2'
                'F_CALL_INCORRECT_POSITION_HAUT', 'F_DEFINITION_UNNECESSARY'
                """
                print(f"Cas pas traite: {errors}")
                cas.add(errors)
        return cas
                
def cas2(c):
    primary_code_errors = c["primary_code_errors"]
    typology_based_code_error = c["typology_based_code_error"]

    # for i in range(len(primary_code_errors)):
    #     regle(primary_code_errors=primary_code_errors[i])
    #     print(f"Fin {i}")
    #     print()
    all = set()
    for i in range(len(typology_based_code_error)):
        print("Liste",typology_based_code_error[i])
        all = all | (regle(typology_based_code_error=typology_based_code_error[i]))
        print(f"Fin {i}")
    return all