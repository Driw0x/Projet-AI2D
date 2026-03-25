import pandas as pd
import ast_error_detection as aed
import ast
import numpy as np
from ast_error_detection.zang_shasha_distance import distance

# Outils AST

def code_to_ast(code):
    """
    Transforme un code Python sous forme de texte en AST.
    AST = arbre syntaxique abstrait.
    """
    return ast.parse(code)

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

def build_user_exercise_error_dict(df):
    """
    Calcule une seule fois les erreurs uniques pour chaque couple (user, exercice).

    Retour :
    - dict {(id_compte, exercice): [erreurs_uniques]}
    """
    df = df[df["code"] != ""]

    user_ex_errors = {}

    for user_id in np.unique(df["id_compte"]):
        user_df = df[df["id_compte"] == user_id]

        for ex in np.unique(user_df["level_1"]):
            sub = user_df[user_df["level_1"] == ex].reset_index(drop=True)

            if len(sub) > 1:
                errs = []
                for i in range(len(sub) - 1):
                    res = primary_code_error_two_prog(
                        sub["code"].iloc[i],
                        sub["code"].iloc[i + 1]
                    )[1]
                    for err in res:
                        errs.append(err[0])

                errs = list(np.unique(errs))
            else:
                errs = []

            key = (
                int(user_id) if isinstance(user_id, np.generic) else user_id,
                str(ex)
            )
            user_ex_errors[key] = errs

    return user_ex_errors
    
# Construction des dictionnaires

def recherche_echantillon(df, user_ex_errors, user_ex_counts):
    """
    Cherche les users ayant plus de 5 tentatives sur au moins un exercice.

    Retour :
    - u : liste des users retenus
    - c : nb de users par exercice
    - e : nb d'apparitions par erreur
    - ex_e : nb d'apparitions par couple (exercice, erreur)
    - ex_u : users par exercice
    """
    df = df[df["code"] != ""]

    if user_ex_errors is None:
        user_ex_errors = build_user_exercise_error_dict(df)

    if user_ex_counts is None:
        user_ex_counts = build_user_exercise_count_dict(df)

    u = []
    c = {}
    e = {}
    ex_e = {}
    ex_u = {}

    all_ex = [str(ex) for ex in np.unique(df["level_1"])]
    all_users = [
        int(user_id) if isinstance(user_id, np.generic) else user_id
        for user_id in np.unique(df["id_compte"])
    ]

    for ex in all_ex:
        ex_u[ex] = []

    for user_id in all_users:
        print(f"User {user_id}")
        u_add = False

        for ex in all_ex:
            key = (user_id, ex)
            nb_tentatives = user_ex_counts.get(key, 0)

            if nb_tentatives > 5:
                erreurs = user_ex_errors.get(key, [])

                ex_u[ex].append(user_id)
                u_add = True

                if ex not in c:
                    c[ex] = 0
                c[ex] += 1

                for err in erreurs:
                    if err not in e:
                        e[err] = 0
                    if (ex, err) not in ex_e:
                        ex_e[(ex, err)] = 0

                    ex_e[(ex, err)] += 1
                    e[err] += 1

                print(f"Exercice {ex}")
                print(f"Erreur: {erreurs}")

        if u_add:
            u.append(user_id)

        print()

    print(u)
    print(c)
    print(e)
    print(ex_e)
    print(ex_u)

    return u, c, e, ex_e, ex_u

class ZSSNode:
    """
    Noeud simple compatible avec l'implémentation Zhang-Shasha utilisée ici.
    """

    def __init__(self, label, children=None, path=None):
        self.label = label
        self.children = children if children is not None else []
        self._path = path if path is not None else [label]

    def get_path(self):
        return self._path


def get_ast_children(node):
    """
    Retourne les vrais enfants AST d'un noeud Python.

    Paramètres
    ----------
    node : ast.AST

    Retour
    ------
    list
        Liste des enfants AST.
    """
    children = []

    for field_name, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            children.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    children.append(item)

    return children


def ast_to_zss_tree(node, path=None, index=0):
    """
    Convertit récursivement un AST Python en arbre compatible Zhang-Shasha.

    Paramètres
    ----------
    node : ast.AST
    path : list ou None
    index : int

    Retour
    ------
    ZSSNode
        Racine de l'arbre converti.
    """
    label = type(node).__name__

    if path is None:
        current_path = [f"{label}[{index}]"]
    else:
        current_path = path + [f"{label}[{index}]"]

    children = []
    ast_children = get_ast_children(node)

    for i, child in enumerate(ast_children):
        children.append(ast_to_zss_tree(child, current_path, i))

    return ZSSNode(label=label, children=children, path=current_path)


def get_children(node):
    """
    Retourne les enfants d'un noeud ZSSNode.

    Paramètres
    ----------
    node : ZSSNode

    Retour
    ------
    list
        Liste des enfants du noeud.
    """
    return node.children


def code_to_zss_tree(code):
    """
    Transforme un code Python en arbre compatible Zhang-Shasha.

    Paramètres
    ----------
    code : str

    Retour
    ------
    ZSSNode
        Racine de l'arbre converti.
    """
    py_ast = ast.parse(code)
    return ast_to_zss_tree(py_ast)


def build_dataset(
    dfo,
    ex_u,
    user_col="id_compte",
    ex_col="level_1",
    code_col="code"
):
    """
    Construit un dataset de transitions entre tentatives.

    Paramètres
    ----------
    dfo : pandas.DataFrame
        Données des tentatives.
    ex_u : dict
        {exercice: [users]}
    user_col : str
    ex_col : str
    code_col : str

    Retour
    ------
    pandas.DataFrame
        Dataset avec id, exercice, t, t_plus_1, distance_zss,
        code_t et code_t_plus_1.
    """
    df = dfo[dfo[code_col].notna() & (dfo[code_col] != "")].copy()

    couples_valides = {
        (user_id, ex)
        for ex, users in ex_u.items()
        for user_id in users
    }

    df = df[df[[user_col, ex_col]].apply(tuple, axis=1).isin(couples_valides)].copy()

    lignes = []

    for (user_id, ex), group in df.groupby([user_col, ex_col], sort=False):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        for i in range(len(group) - 1):
            code_t = group.loc[i, code_col]
            code_t_plus_1 = group.loc[i + 1, code_col]

            try:
                tree_t = code_to_zss_tree(code_t)
                tree_t_plus_1 = code_to_zss_tree(code_t_plus_1)

                dist, _ = distance(tree_t, tree_t_plus_1, get_children)

            except Exception as err:
                print(f"Erreur distance user={user_id}, ex={ex}, t={i+1}: {err}")
                dist = np.nan

            ligne = {
                "id": user_id,
                "exercice": ex,
                "t": i + 1,
                "t_plus_1": i + 2,
                "distance_zss": dist,
                "code_t": code_t,
                "code_t_plus_1": code_t_plus_1
            }

            lignes.append(ligne)

    return pd.DataFrame(lignes)