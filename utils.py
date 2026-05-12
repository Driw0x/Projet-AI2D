import pandas as pd
import ast_error_detection as aed
import ast
import numpy as np
import json
import textwrap
from tqdm import tqdm, trange
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
    tree = code_to_ast(code)
    if tree is None:
        return None
    return Wrapper(tree)

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

# Cas 1 / Pre-calcul : construction commune des transitions t -> t+1
import warnings

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


def compare_transition(
    code_t,
    code_t_1,
    zss_cache=None,
    ast_solutions=None,
    include_code_errors=False,
    include_solution_scores=False,
    context=""
):
    """
    Compare deux codes consécutifs t -> t+1.
    """
    if zss_cache is None:
        zss_cache = {}
    if ast_solutions is None:
        ast_solutions = []

    result = {
        "distance_zss": np.nan,
        "ops": {},
        "primary_code_errors_score": np.nan,
        "primary_code_errors": [],
        "typology_based_code_error_score": np.nan,
        "typology_based_code_error": {},
        "score_t_solution": np.nan,
        "score_t_plus_1_solution": np.nan,
        "progression_solution": np.nan,
    }

    # Distance AST/ZSS entre t et t+1
    try:
        tree_t = get_zss_tree(code_t, zss_cache)
        tree_t_1 = get_zss_tree(code_t_1, zss_cache)

        if tree_t is not None and tree_t_1 is not None:
            result["distance_zss"], result["ops"] = distance(tree_t, tree_t_1, get_children)
    except Exception as err:
        tqdm.write(f"Erreur distance {context}: {err}")

    # Erreurs entre t et t+1, utile pour cas1/cas2
    if include_code_errors:
        try:
            with warnings.catch_warnings(record=True) as warns:
                warnings.simplefilter("always", SyntaxWarning)
                primary = primary_code_error_two_prog(code_t, code_t_1)
                typology = prog_vs_answer(code_t, [code_t_1])

                for warn in warns:
                    tqdm.write(f"Warning {context}: {warn.message}")

            result["primary_code_errors_score"] = primary[0]
            result["primary_code_errors"] = primary[1]
            result["typology_based_code_error_score"] = typology[0]
            result["typology_based_code_error"] = typology[1]
        except Exception as err:
            tqdm.write(f"Erreur comparaison t->t+1 {context}: {err}")
            result["primary_code_errors_score"] = 0
            result["primary_code_errors"] = []
            result["typology_based_code_error_score"] = 0
            result["typology_based_code_error"] = {}

    # Scores par rapport aux solutions, utile pour pre_calcul/classification
    if include_solution_scores:
        ast_t = code_to_ast(code_t)
        ast_t_1 = code_to_ast(code_t_1)

        try:
            comp_t_sol = prog_vs_answer(ast_t, ast_solutions) if ast_t is not None and ast_solutions else [0, {}]
        except Exception as err:
            tqdm.write(f"Erreur comparaison t->solution {context}: {err}")
            comp_t_sol = [0, {}]

        try:
            comp_t1_sol = prog_vs_answer(ast_t_1, ast_solutions) if ast_t_1 is not None and ast_solutions else [0, {}]
        except Exception as err:
            tqdm.write(f"Erreur comparaison t+1->solution {context}: {err}")
            comp_t1_sol = [0, {}]

        result["score_t_solution"] = extract_score(comp_t_sol)
        result["score_t_plus_1_solution"] = extract_score(comp_t1_sol)
        result["progression_solution"] = result["score_t_plus_1_solution"] - result["score_t_solution"]

    return result


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


def cas1_x_y(data, x, y, zss_cache=None):
    if x < 0 or y < 0 or x >= len(data) or y >= len(data):
        return np.nan, {}, (0, []), [0, {}]

    result = compare_transition(
        data.loc[x, "code"],
        data.loc[y, "code"],
        zss_cache=zss_cache,
        include_code_errors=True,
        include_solution_scores=False,
        context=f"t={x + 1}, t'={y + 1}",
    )

    primary = (result["primary_code_errors_score"], result["primary_code_errors"])
    typology = [result["typology_based_code_error_score"], result["typology_based_code_error"]]
    return result["distance_zss"], result["ops"], primary, typology


def cas1(dfo, solution_df):
    """
    Construit le dataset des comparaisons t -> t+1 avec les erreurs AED.
    """
    dataset = build_transition_dataset(
        dfo,
        solution_df,
        include_code_errors=True,
        include_solution_scores=False,
        include_temps=False,
        include_status=False,
        include_codes=True,
        save_path=None,
    )

    # Compatibilité avec l'ancien nom de colonne.
    dataset["dist_zss"] = dataset["distance_zss"]

    ordered_cols = [
        "id",
        "exercice",
        "type_exercice",
        "t",
        "t_plus_1",
        "dist_zss",
        "distance_zss",
        "ops",
        "code_t",
        "code_t_1",
        "primary_code_errors_score",
        "primary_code_errors",
        "typology_based_code_error_score",
        "typology_based_code_error",
    ]
    dataset = dataset[[col for col in ordered_cols if col in dataset.columns]]

    save_dataset_to_json(dataset, "cas1.json")
    return dataset

# Cas 2
def regle(ops=None, primary_code_errors=None, typology_based_code_error=None, ):
    """
    Affiche une interpretation textuelle simple des erreurs detectees.
    """
    def path_trad(path):
        path.reverse()
        f = 1
        morceaux = []

        for p in path[1:]:
            if p == "For":
                morceaux.append(f"dans la {f}e boucle for qui se trouve")
                f += 1
            elif p.startswith("For"):
                idx = int(p.split('[')[-1].split(']')[0]) + 1
                morceaux.append(f"dans la {idx}e boucle for qui se trouve")
                f = 0
            elif p.startswith("Call"):
                morceaux.append(f"dans l'argument de la fonction {p.split(': ')[-1]}")
            elif p.startswith("Module"):
                morceaux.append("dans le code")

        return " ".join(morceaux).strip()

    phrases = []
    cas = set()

    if ops is not None:
        return phrases, cas
    
    elif primary_code_errors is not None:
        """
        Pas exploitable: 'MISSING_VARIABLE', 'UNNECESSARY_VAR', 'VARIABLE_MISMATCH', 
        'INCORRECT_STATEMENT_POSITION_FOR', 'MISSING_CONST_VALUE', 'UNNECESSARY_CONST_VALUE',
        'NODE_TYPE_MISMATCH', 
 
        'UNNECESSARY_ASSIGN_STATEMENT': Assignation var inutile
        'MISSING_ASSIGN_STATEMENT': Ajout assignation val à var
        'INCORRECT_OPERATION_IN_ASSIGN' ???
        'UNNECESSARY_ARGUMENT': suppression d'un argument d'une fonction (potentiellement la fonction aussi)

        Liste à traiter: 'UNNECESSARY_RETURN_IN_FUNCTION', 'INCORRECT_OPERATION_IN_CONDITION', 'MISSING_ARGUMENT', 'INCORRECT_STATEMENT_POSITION_ASSIGN'
        """
        for errors in primary_code_errors:
            path = errors[-1].split(" > ")
            end = path_trad(path)

            match errors[0]:
                case "MISSING_CALL_STATEMENT" | "MISSING_FOR_LOOP":
                    if path[-1].startswith(errors[1]):
                        phrases.append(f"Ajout d'un appel à {errors[1].split(' ')[-1].lower()} {end}".strip())
                    else:
                        # Ce cas sera traité par d'autre erreur
                        pass

                case "UNNECESSARY_CALL_STATEMENT":
                    if len(errors) == 3:
                        phrases.append(f"Suppression d'un appel à {errors[1].split(': ')[-1]} {end}".strip())
                    else:
                        phrases.append(f"Appel à {errors[2].split(': ')[-1]} sur la position de l'appel à {errors[1].split(': ')[-1]} {end}".strip())
                
                case "UNNECESSARY_FUNCTION":
                    """
                    Cas critique:
                    "code_t": "def hexagone():\n    for k in range(6):\n        avancer(2)\n        tourner(60)\n    tourner(60)\n",
                    "code_t_1": "def hexagone():\n    avancer(2)\n    tourner(60)\nhexagone()\n",
                    "UNNECESSARY_FUNCTION",
                    "hexagone",
                    "Module > Function: hexagone[0]"
                    """
                    phrases.append(f"Suppression de la fonction {errors[1]}")

                case 'INCORRECT_STATEMENT_POSITION_IF':
                    phrases.append(f"Changement de la position du if {end}".strip())

                case 'MISSING_OPERATION':
                    phrases.append(f"Ajout de l'opération {errors[1].split(': ')[-1]} {end}".strip())
                
                case 'UNNECESSARY_VARIABLE':
                    if not path[-2].startswith("Condition:"):
                        phrases.append(f"Probleme d'appel de fonction {end}".strip())

                case 'UNNECESSARY_OPERATION':
                    if len(errors) == 3:
                        phrases.append(f"Suppression de l'opération {errors[1]} {end}".strip())
                    else:
                        phrases.append(f"Changement de l'opération {errors[1]} en {errors[2]} {end}".strip())
                
                case 'MISSING_FUNCTION_DEFINITION':
                    phrases.append(f"Ajout de {errors[1].split(': ')[-1]} {end}".strip())

                case 'INCORRECT_STATEMENT_POSITION_CALL':
                    phrases.append(f"Changement de position de l'appel à {errors[1].lower()} {end}".strip())
                
                case 'UNNECESSARY_FOR_LOOP':
                    phrases.append(f"Supression de la boucle for {end}".strip())

                case 'CONST_VALUE_MISMATCH':
                    phrases.append(f"Changement de la constante {errors[1].split(': ')[-1]} en {errors[2].split(': ')[-1]} {end}".strip())
                
                case 'MISSING_IF_STATEMENT':
                    phrases.append(f"Ajout d'une condition {end}".strip())

                case 'UNNECESSARY_CONDITIONAL':
                    phrases.append(f"Suppression d'une condition {end}".strip())
                
                case 'INCORRECT_STATEMENT_POSITION_FUNCTION':
                    phrases.append(f"Changement de la position de l'appel à {errors[1].lower()} {end}".strip())
                
                case 'MISSING_VARIABLE' | 'UNNECESSARY_VAR' | 'VARIABLE_MISMATCH'| 'INCORRECT_STATEMENT_POSITION_FOR' | 'MISSING_CONST_VALUE' | 'UNNECESSARY_CONST_VALUE' | 'NODE_TYPE_MISMATCH' | 'UNNECESSARY_ASSIGN_STATEMENT' | 'MISSING_ASSIGN_STATEMENT' | 'INCORRECT_OPERATION_IN_ASSIGN' | 'UNNECESSARY_ARGUMENT':
                    pass
                case _:
                    cas.add(errors[0])
        
    elif typology_based_code_error is not None:
        for errors in typology_based_code_error:
            if errors.startswith("F_CALL_MISSING"):
                phrases.append(f"Ajout d'un appel à la fonction {errors.split('_')[-1].lower()}")
                continue
            elif errors == "F_CALL_UNNECESSARY":
                continue
            elif errors.startswith("F_CALL_UNNECESSARY"):
                phrases.append(f"Suppression d'un appel à la fonction {errors.split('_')[-1].lower()}")
                continue
            elif errors.startswith("F_CALL_INCORRECT_POSITION"):
                # Position exacte dans la partie primary
                phrases.append(f"Changement de position de la fonction {errors.split('_')[-1].lower()}")
                continue
            elif errors == "F_CALL_PRINT_ERROR_ARG":
                phrases.append(f"Changement d'argument dans la fonction {errors.split('_')[-3].lower()}")
            elif (errors.startswith("F_CALL") and errors.endswith("_ERROR")) :
                phrases.append(f"Changement d'argument dans la fonction {errors.split('_')[-2].lower()}")
                continue
            elif errors.startswith("F_DEFINITION"):
                if errors.endswith("MISSING"):
                    # L'info de la fonction qui a été ajouté sera dans la partie primary
                    phrases.append("Ajout d'une fonction")
                    continue
                elif errors.endswith("UNNECESSARY"):
                    phrases.append("Suppression d'une fonction")
                    continue
                else:
                    cas.add(errors)
                    continue
            match errors:
                case "LO_FOR_MISPLACED":
                    phrases.append("Modification d'une boucle for (potentiellement un ajout)")
                case "LO_FOR_MISSING":
                    phrases.append("Ajout d'une boucle for")
                case 'LO_FOR_UNNECESSARY':
                    phrases.append("Suppression d'une boucle for")
                case 'LO_BODY_MISSING_NOT_PRESENT_ANYWHERE':
                    phrases.append("Modification du corps d'une boucle for (potentiellement supprimée)")
                case 'LO_FOR_NUMBER_ITERATION_ERROR':
                    phrases.append("Différence sur le nombre d'itération dans une boucle")
                case 'LO_FOR_NUMBER_ITERATION_ERROR_UNDER2':
                    phrases.append("Différence sur le nombre d'itération < 2 dans une boucle")
                case 'LO_BODY_MISPLACED':
                    phrases.append("Modification de l'ordre des appels dans le code")
                case 'CS_MISSING':
                    # Pas sur
                    phrases.append("Ajout d'une structure de comparaison")
                case 'EXP_ERROR_ASSIGNMENT_MISSING':
                    phrases.append("Ajout d'une assigration de valeur à une variable")
                case _:
                    cas.add(errors)

            """
            Cas à traiter avec primary de préférence: EXP_ERROR_OPERATOR, LO_BODY_ERROR,
            EXP_ERROR_OPERATION, EXP_ERROR_OPERANDS

            Cas unitaire sur 2025 (à revoir): 'EXP_ERROR_ASSIGNMENT_UNNECESSARY'
            """
    return phrases, cas
                
def cas2(c):
    c2 = c.copy()

    primary_texts = []
    typology_texts = []

    for i in trange(len(c2),
                    desc="Génération des commentaires",
                    leave=True):
        primary_val = c2.loc[i, "primary_code_errors"] if "primary_code_errors" in c2.columns else None
        typology_val = c2.loc[i, "typology_based_code_error"] if "typology_based_code_error" in c2.columns else None

        p_phrases, _ = regle(primary_code_errors=primary_val)
        t_phrases, _ = regle(typology_based_code_error=typology_val)

        primary_texts.append(p_phrases)
        typology_texts.append(t_phrases)

    c2["primary_code_errors_text"] = primary_texts
    c2["typology_based_code_error_text"] = typology_texts
    
    save_dataset_to_json(c2, "cas2.json")

    return c2

# Classification
def extract_score(comp):
    """
    Extrait le score depuis une structure du type [score, details].
    Retourne NaN si le format est invalide.
    """
    try:
        if isinstance(comp, (list, tuple)) and len(comp) > 0:
            return float(comp[0])
        return np.nan
    except Exception:
        return np.nan


def remove_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    Retire les valeurs extrêmes avec la règle IQR.
    """
    s = series.dropna()
    if len(s) == 0:
        return s

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return s

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = s[(s >= lower) & (s <= upper)]
    return filtered if len(filtered) > 0 else s


def compute_tertile_thresholds(series: pd.Series, fallback=(0.33, 0.66)):
    """
    Calcule les seuils tertiles après retrait des outliers.
    """
    s = remove_outliers_iqr(series)

    if len(s) == 0:
        return fallback

    t1 = s.quantile(1 / 3)
    t2 = s.quantile(2 / 3)
    return t1, t2


def compute_dynamic_thresholds(user_stats: pd.DataFrame) -> dict:
    """
    Calcule tous les seuils dynamiques à partir des stats agrégées.
    """
    thresholds = {}

    thresholds["progress"] = compute_tertile_thresholds(
        user_stats["mean_progress"],
        fallback=(0.0, 1.0)
    )

    thresholds["zss"] = compute_tertile_thresholds(
        user_stats["mean_zss"],
        fallback=(3.0, 8.0)
    )

    thresholds["tentatives"] = compute_tertile_thresholds(
        user_stats["max_tentative"],
        fallback=(3, 6)
    )

    thresholds["temps"] = compute_tertile_thresholds(
        user_stats["mean_delta_temps"],
        fallback=(30, 120)
    )

    return thresholds


def profil_progression_dynamic(mean_progress, thresholds):
    """
    Profil de progression avec seuils dynamiques.
    """
    if pd.isna(mean_progress):
        return "inconnu"

    t1, t2 = thresholds["progress"]

    if mean_progress < 0:
        return "regression"
    if mean_progress <= t1:
        return "stagnation"
    if mean_progress <= t2:
        return "progression_lente"
    return "progression_rapide"


def profil_modification_dynamic(mean_zss, thresholds):
    """
    Profil de modification avec seuils dynamiques.
    """
    if pd.isna(mean_zss):
        return "inconnu"

    t1, t2 = thresholds["zss"]

    if mean_zss <= t1:
        return "petit_ajusteur"
    if mean_zss <= t2:
        return "modificateur_modere"
    return "gros_restructurateur"


def profil_tentatives_dynamic(nb_tentatives, thresholds):
    """
    Profil de tentatives avec seuils dynamiques.
    """
    if pd.isna(nb_tentatives):
        return "inconnu"

    t1, t2 = thresholds["tentatives"]

    if nb_tentatives <= t1:
        return "resolution_rapide"
    if nb_tentatives <= t2:
        return "resolution_moyenne"
    return "resolution_longue"


def profil_temps_dynamic(mean_temps, nb_temps_valides, thresholds):
    """
    Profil de temps avec seuils dynamiques.
    Ignore le temps s'il n'est pas renseigné.
    """
    if nb_temps_valides == 0 or pd.isna(mean_temps):
        return "non_renseigne"

    t1, t2 = thresholds["temps"]

    if mean_temps <= t1:
        return "rapide"
    if mean_temps <= t2:
        return "modere"
    return "lent"


def profil_reussite_exercice(reussite_finale):
    if pd.isna(reussite_finale):
        return "inconnu"
    return "reussi" if reussite_finale == 1 else "non_reussi"


def profil_taux_reussite(taux_reussite):
    if pd.isna(taux_reussite):
        return "inconnu"

    if taux_reussite >= 0.75:
        return "forte_reussite"
    if taux_reussite >= 0.5:
        return "reussite_moyenne"
    return "faible_reussite"


def classe_finale_from_profils(profil_prog, profil_modif, profil_temps, profil_tent, profil_reussite=None):
    """
    Déduit une classe finale à partir des profils.

    Version moins restrictive : avant, presque toutes les règles étaient dans
    `if profil_temps == "non_renseigne"`, donc dès que le temps était présent,
    la fonction retournait très souvent `profil_intermediaire`.

    Ici, le temps sert comme information complémentaire, mais ne bloque plus
    toute la classification.
    """
    bonne_reussite = profil_reussite in [None, "inconnu", "reussi", "forte_reussite", "reussite_moyenne"]
    mauvaise_reussite = profil_reussite in ["non_reussi", "faible_reussite"]
    temps_rapide_ou_absent = profil_temps in ["rapide", "non_renseigne"]
    temps_lent = profil_temps == "lent"

    # 1) Cas en difficulté : priorité aux échecs / faibles réussites.
    if mauvaise_reussite:
        if profil_prog in ["stagnation", "regression"]:
            return "bloque"
        if profil_modif == "gros_restructurateur":
            return "explorateur_chaotique"
        if profil_tent == "resolution_longue" or temps_lent:
            return "en_difficulte"
        return "fragile"

    # 2) Bons profils : progression efficace avec peu de changements.
    if (
        bonne_reussite
        and profil_prog == "progression_rapide"
        and profil_modif == "petit_ajusteur"
        and profil_tent == "resolution_rapide"
        and temps_rapide_ou_absent
    ):
        return "expert_progressif"

    if (
        bonne_reussite
        and profil_prog in ["progression_rapide", "progression_lente"]
        and profil_modif in ["petit_ajusteur", "modificateur_modere"]
        and profil_tent in ["resolution_rapide", "resolution_moyenne"]
    ):
        return "reviseur_methodique"

    # 3) Progression avec grosses modifications : il restructure beaucoup, mais avance.
    if profil_prog in ["progression_rapide", "progression_lente"] and profil_modif == "gros_restructurateur":
        if profil_tent == "resolution_longue" or temps_lent:
            return "restructurateur_lent"
        return "gros_restructurateur_productif"

    # 4) Peu ou pas de progression.
    if profil_prog in ["stagnation", "regression"]:
        if profil_modif == "gros_restructurateur":
            return "explorateur_chaotique"
        if profil_tent == "resolution_longue" or temps_lent:
            return "bloque"
        return "stagne_mais_corrige"

    # 5) Cas intermédiaires plus informatifs que `profil_intermediaire`.
    if profil_prog == "progression_lente" and profil_tent == "resolution_longue":
        return "perseverant_lent"

    if profil_modif == "petit_ajusteur" and profil_tent == "resolution_longue":
        return "petits_pas_nombreux"

    if profil_modif == "modificateur_modere":
        return "apprenant_regulier"

    return "profil_intermediaire"


def distribution_classes(stats, col="classe"):
    """
    Retourne la distribution des classes en nombre et en pourcentage.
    Utile pour vérifier si la classification est trop uniforme.
    """
    counts = stats[col].value_counts(dropna=False)
    pct = stats[col].value_counts(normalize=True, dropna=False).mul(100).round(2)

    return (
        pd.DataFrame({"nb": counts, "%": pct})
        .reset_index()
        .rename(columns={"index": col})
    )


def diagnostic_classification(stats):
    """
    Diagnostic rapide pour comprendre pourquoi les classes sont concentrées.
    """
    colonnes = [
        "profil_progression",
        "profil_modification",
        "profil_tentatives",
        "profil_temps",
        "profil_reussite",
        "classe",
    ]

    return {
        col: distribution_classes(stats, col)
        for col in colonnes
        if col in stats.columns
    }


def pre_calcul(dfo, solution_df):
    """
    Construit un dataset allégé pour la classification.
    """
    dataset = build_transition_dataset(
        dfo,
        solution_df,
        include_code_errors=False,
        include_solution_scores=True,
        include_temps=True,
        include_status=True,
        include_codes=False,
        save_path="pre_calcul.json",
    )

    cols = [
        "id",
        "exercice",
        "type_exercice",
        "t",
        "t_plus_1",
        "distance_zss",
        "score_t_solution",
        "score_t_plus_1_solution",
        "progression_solution",
        "temps_t",
        "temps_t_plus_1",
        "delta_temps",
        "statut_t",
        "statut_t_plus_1",
        "reussite_finale_exercice",
    ]
    return dataset[[col for col in cols if col in dataset.columns]]


def aggregate_classification_stats(dataset, group_cols, success_mode):
    """
    Agrège les métriques nécessaires à la classification.

    success_mode:
    - "user" : calcule un taux de réussite par utilisateur.
    - "exercise" : garde la réussite finale du couple user/exercice.
    """
    df = dataset.copy()
    check_required_columns(
        df,
        [
            "id",
            "exercice",
            "progression_solution",
            "distance_zss",
            "t_plus_1",
            "delta_temps",
            "reussite_finale_exercice",
        ]
    )

    agg_dict = {
        "mean_progress": ("progression_solution", "mean"),
        "mean_zss": ("distance_zss", "mean"),
        "max_tentative": ("t_plus_1", "max"),
        "nb_transitions": ("id", "size"),
        "mean_delta_temps": ("delta_temps", "mean"),
        "nb_temps_valides": ("delta_temps", lambda s: s.notna().sum()),
    }

    if "type_exercice" in df.columns and "exercice" in group_cols:
        agg_dict["type_exercice"] = ("type_exercice", "first")

    stats = df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()

    if success_mode == "user":
        reussite_ex = (
            df.groupby(["id", "exercice"], dropna=False)
            .agg(reussite_finale_exercice=("reussite_finale_exercice", "max"))
            .reset_index()
        )

        success = (
            reussite_ex.groupby("id", dropna=False)
            .agg(
                taux_reussite=("reussite_finale_exercice", "mean"),
                nb_exercices=("exercice", "size"),
                nb_exercices_reussis=("reussite_finale_exercice", "sum"),
            )
            .reset_index()
        )
        stats = stats.merge(success, on="id", how="left")
        stats["profil_reussite"] = stats["taux_reussite"].apply(profil_taux_reussite)

    elif success_mode == "exercise":
        success = (
            df.groupby(group_cols, dropna=False)
            .agg(reussite_finale_exercice=("reussite_finale_exercice", "max"))
            .reset_index()
        )
        stats = stats.merge(success, on=group_cols, how="left")
        stats["profil_reussite"] = stats["reussite_finale_exercice"].apply(profil_reussite_exercice)

    else:
        raise ValueError("success_mode doit être 'user' ou 'exercise'")

    return stats


def apply_dynamic_profiles(stats):
    """
    Ajoute les profils dynamiques et la classe finale.
    """
    thresholds = compute_dynamic_thresholds(stats)

    stats = stats.copy()
    stats["profil_progression"] = stats["mean_progress"].apply(
        lambda x: profil_progression_dynamic(x, thresholds)
    )
    stats["profil_modification"] = stats["mean_zss"].apply(
        lambda x: profil_modification_dynamic(x, thresholds)
    )
    stats["profil_tentatives"] = stats["max_tentative"].apply(
        lambda x: profil_tentatives_dynamic(x, thresholds)
    )
    stats["profil_temps"] = stats.apply(
        lambda row: profil_temps_dynamic(
            row["mean_delta_temps"],
            row["nb_temps_valides"],
            thresholds,
        ),
        axis=1,
    )

    stats["classe"] = stats.apply(
        lambda row: classe_finale_from_profils(
            row["profil_progression"],
            row["profil_modification"],
            row["profil_temps"],
            row["profil_tentatives"],
            row["profil_reussite"],
        ),
        axis=1,
    )

    return stats, thresholds


def build_user_classification(dataset):
    """
    Classification globale par utilisateur.
    """
    stats = aggregate_classification_stats(
        dataset,
        group_cols=["id"],
        success_mode="user",
    )
    return apply_dynamic_profiles(stats)


def build_user_exercice_classification(dataset):
    """
    Classification par couple (utilisateur, exercice).
    """
    stats = aggregate_classification_stats(
        dataset,
        group_cols=["id", "exercice"],
        success_mode="exercise",
    )
    return apply_dynamic_profiles(stats)
