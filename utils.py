import pandas as pd
import ast_error_detection as aed
import ast
import numpy as np
import json
import textwrap
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
    tentatives = pd.json_normalize(traj_long["tentatives"])

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
    
def regle(primary_code_error):
    """
    Affiche une interpretation textuelle simple des erreurs detectees.
    """
    def extract_function_name(s):
        """
        Extrait le nom de la fonction à partir du contexte.
        """
        return s.split(":")[-1].strip()
    
    for i in primary_code_error[1]:
        erreur = i[0]
        context = i[-1].split(" > ")

        print(erreur)

        match erreur:
            case 'CONST_VALUE_MISMATCH':
                print(f"\tConstante en argument pour la fonction {extract_function_name(context[-2])} incorrecte.")

            case 'MISSING_CONST_VALUE':
                print(f"\tManque l'argument à la fonction {extract_function_name(context[-2])} dans le programme")

            case 'MISSING_CALL_STATEMENT' | 'MISSING_FOR_LOOP':
                print(f"\tManque l'appel à la fonction {extract_function_name(context[-1])} dans le programme")

            case _ if 'UNNECESSARY' in erreur:
                print(f"\tAppel inutile à la fonction {extract_function_name(context[-1])} dans le programme")

            case _:
                print("\tRetour pas encore prise en charge")

def comparaison_tentative_solution(df):
    """
    Compare toutes les tentatives d'un exercice à la dernière tentative,
    consideree ici comme la solution de reference.
    """
    prog = []
    p = df.reset_index(drop=True)

    # recupère tous les codes
    for i in range(len(p)):
        prog.append(p.iloc[i]["code"])

    # dernière tentative = reponse de reference
    answer = p.iloc[len(p) - 1]["code"]
    ast_answer = code_to_ast(answer)

    # compare chaque tentative à la reponse
    for i in range(len(prog) - 1):
        print(f"Tentative {i + 1}: ")

        if prog[i]:
            ast_p = code_to_ast(prog[i])
            regle(primary_code_error_two_prog(ast_p, ast_answer))
        else:
            print("Code vide")

def analyse_user(id_compte, df):
    """
    Analyse tous les exercices d'un utilisateur.
    Pour chaque exercice :
    - compare les tentatives si plusieurs essais
    - indique reussite directe ou abandon si un seul essai
    """
    u = df[(df["id_compte"] == id_compte)]
    ex = np.unique(u["level_1"])

    for e in ex:
        print(f"Exercice {e}:")
        p = u[u["level_1"] == e]

        if len(p) > 1:
            comparaison_tentative_solution(p)
        else:
            print("Qu'un seul essai")
            if p["statut"].iloc[0] == 1:
                print("Il a reussi du premier coup")
            else:
                print("Il a abandonne dès le premier essai")
        print()

# Pre calcul

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
    return Wrapper(ast.parse(code))

def get_children(node):
    """
    Retourne les enfants d'un noeud compatible Zhang-Shasha.
    """
    return node.children

# Construction des dictionnaires

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

def build_datasets(dfo, solution_df, save_json=True):
    """
    Construit:
    - le dataset des transitions successives (t, t+1)
    - le dataset du saut maximal ZSS par (user, exercice)

    Retour:
        (dataset_successif, dataset_max_zss)
    """
    df = dfo[dfo["code"].notna() & (dfo["code"] != "")].copy()

    couples = couples_valides(dfo)
    df = df[df[["id_compte", "level_1"]].apply(tuple, axis=1).isin(couples)].copy()

    solution_dict = build_solution_dict(solution_df)

    lignes_successif = []
    lignes_max_zss = []

    for (user_id, ex), group in df.groupby(["id_compte", "level_1"], sort=False):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        sol_info = solution_dict.get(ex)
        if sol_info is not None:
            correct_codes = sol_info["correctCodes"]
            exercise_type = sol_info["exerciseType"]

            ast_solutions = []
            for sol in correct_codes:
                ast_sol = code_to_ast(sol)
                if ast_sol is not None:
                    ast_solutions.append(ast_sol)
        else:
            correct_codes = []
            exercise_type = np.nan
            ast_solutions = []

        # Cache local pour éviter de parser/recalculer plusieurs fois
        ast_cache = {}
        zss_cache = {}

        def get_ast(code):
            if code not in ast_cache:
                ast_cache[code] = code_to_ast(code)
            return ast_cache[code]

        def get_zss_tree(code):
            if code not in zss_cache:
                try:
                    zss_cache[code] = code_to_zss_node(code)
                except Exception:
                    zss_cache[code] = None
            return zss_cache[code]

        best_row = None
        best_dist = -np.inf

        for i in range(len(group) - 1):
            for j in range(i + 1, len(group)):
                code_t = group.loc[i, "code"]
                code_t_plus = group.loc[j, "code"]

                ast_t = get_ast(code_t)
                ast_t_plus = get_ast(code_t_plus)

                # distance ZSS
                try:
                    tree_t = get_zss_tree(code_t)
                    tree_t_plus = get_zss_tree(code_t_plus)

                    if tree_t is not None and tree_t_plus is not None:
                        dist, _ = distance(tree_t, tree_t_plus, get_children)
                    else:
                        dist = np.nan
                except Exception as err:
                    print(f"Erreur distance user={user_id}, ex={ex}, t={i+1}, t_plus={j+1}: {err}")
                    dist = np.nan

                # comparaison t -> t_plus
                try:
                    if ast_t is not None and ast_t_plus is not None:
                        comp_tt1 = primary_code_error_two_prog(ast_t, ast_t_plus)
                    else:
                        comp_tt1 = [0, []]
                except Exception as err:
                    print(f"Erreur comparaison t->t_plus user={user_id}, ex={ex}, t={i+1}, t_plus={j+1}: {err}")
                    comp_tt1 = [0, []]

                try:
                    nb_err_tt1 = len(comp_tt1[1])
                except Exception:
                    nb_err_tt1 = 0

                # comparaison t -> solution
                try:
                    if ast_t is not None and ast_solutions:
                        comp_t = prog_vs_answer(ast_t, ast_solutions)
                    else:
                        comp_t = [0, {}]
                except Exception as err:
                    print(f"Erreur comparaison solution user={user_id}, ex={ex}, t={i+1}: {err}")
                    comp_t = [0, {}]

                # comparaison t_plus -> solution
                try:
                    if ast_t_plus is not None and ast_solutions:
                        comp_t_plus = prog_vs_answer(ast_t_plus, ast_solutions)
                    else:
                        comp_t_plus = [0, {}]
                except Exception as err:
                    print(f"Erreur comparaison solution user={user_id}, ex={ex}, t={j+1}: {err}")
                    comp_t_plus = [0, {}]

                row = {
                    "id": user_id,
                    "exercice": ex,
                    "type_exercice": exercise_type,
                    "t": i + 1,
                    "t_plus_1": j + 1,
                    "distance_zss": dist,
                    "comparaison_t_t_plus_1": comp_tt1,
                    "nb_erreurs_t_t_plus_1": nb_err_tt1,
                    "comparaison_t_solution": comp_t,
                    "comparaison_t_plus_1_solution": comp_t_plus,
                    "code_t": code_t,
                    "code_t_plus_1": code_t_plus,
                    "solution": list(correct_codes)
                }

                # Dataset successif : seulement j = i+1
                if j == i + 1:
                    lignes_successif.append(row)

                # Dataset max_zss : on garde le meilleur (t, t+i)
                dist_cmp = -np.inf if pd.isna(dist) else dist
                if dist_cmp > best_dist:
                    best_dist = dist_cmp
                    best_row = row

        if best_row is not None:
            lignes_max_zss.append(best_row)

    dataset_successif = pd.DataFrame(lignes_successif)
    dataset_max_zss = pd.DataFrame(lignes_max_zss)

    if save_json:
        save_dataset_to_json(dataset_successif, "dataset.json")
        save_dataset_to_json(dataset_max_zss, "dataset_max_zss.json")

    return dataset_successif, dataset_max_zss

# Sauvegarde

def clean_value(v):
    """
    Nettoie une valeur pour la rendre JSON compatible.
    """
    # None direct
    if v is None:
        return None

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