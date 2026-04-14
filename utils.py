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

# Cas 1: transformation t->t+1
import warnings
def cas1_x_y(data, x, y, zss_cache=None):
    """
    Compare le code x et y.
    """
    if x < 0 or y < 0 or x >= len(data) or y >= len(data):
        return np.nan, {}, (0, []), [0, {}]

    if zss_cache is None:
        zss_cache = {}

    code_x = data.loc[x, "code"]
    code_y = data.loc[y, "code"]

    # comparaison x -> y
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", SyntaxWarning)
            primary = primary_code_error_two_prog(code_x, code_y)
            typology = prog_vs_answer(code_x, [code_y])

            for warn in w:
                tqdm.write(f"Warning t={x+1}, t'={y+1}: {warn.message}")
                tqdm.write("code_x:")
                tqdm.write(code_x)
                tqdm.write("code_y:")
                tqdm.write(code_y)
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

def profil_progression(mean_progress,
                       strong_progress_threshold=1.0,
                       weak_progress_threshold=0.2):
    if pd.isna(mean_progress):
        return "inconnu"
    if mean_progress >= strong_progress_threshold:
        return "progression_rapide"
    if mean_progress >= weak_progress_threshold:
        return "progression_lente"
    if mean_progress >= 0:
        return "stagnation"
    return "regression"


def profil_modification(mean_zss,
                        low_zss_threshold=3.0,
                        high_zss_threshold=8.0):
    if pd.isna(mean_zss):
        return "inconnu"
    if mean_zss <= low_zss_threshold:
        return "petit_ajusteur"
    if mean_zss <= high_zss_threshold:
        return "modificateur_modere"
    return "gros_restructurateur"


def profil_tentatives(nb_tentatives,
                      fast_attempt_threshold=3,
                      many_attempts_threshold=6):
    if pd.isna(nb_tentatives):
        return "inconnu"
    if nb_tentatives <= fast_attempt_threshold:
        return "resolution_rapide"
    if nb_tentatives <= many_attempts_threshold:
        return "resolution_moyenne"
    return "resolution_longue"


def profil_temps(mean_temps,
                 fast_time_threshold=30,
                 medium_time_threshold=120):
    """
    Retourne un profil de temps.
    Si le temps n'est pas renseigné, retourne 'non_renseigne'.
    """
    if pd.isna(mean_temps):
        return "non_renseigne"
    if mean_temps <= fast_time_threshold:
        return "rapide"
    if mean_temps <= medium_time_threshold:
        return "modere"
    return "lent"

def classe_finale_from_profils(profil_prog, profil_modif, profil_temps, profil_tent):
    # Cas où le temps n'est pas disponible : on l'ignore
    if profil_temps == "non_renseigne":
        if (
            profil_prog == "progression_rapide"
            and profil_modif == "petit_ajusteur"
            and profil_tent == "resolution_rapide"
        ):
            return "expert_progressif"

        if (
            profil_prog in ["progression_rapide", "progression_lente"]
            and profil_modif in ["petit_ajusteur", "modificateur_modere"]
            and profil_tent in ["resolution_rapide", "resolution_moyenne"]
        ):
            return "reviseur_methodique"

        if (
            profil_prog in ["progression_rapide", "progression_lente"]
            and profil_modif == "gros_restructurateur"
        ):
            return "gros_restructurateur"

        if (
            profil_prog in ["stagnation", "regression"]
            and profil_tent == "resolution_longue"
        ):
            return "bloque"

        if (
            profil_prog in ["stagnation", "regression"]
            and profil_modif == "gros_restructurateur"
        ):
            return "explorateur_chaotique"

        return "profil_intermediaire"

    # Cas où le temps est disponible : on l'utilise
    if (
        profil_prog == "progression_rapide"
        and profil_modif == "petit_ajusteur"
        and profil_tent == "resolution_rapide"
        and profil_temps == "rapide"
    ):
        return "expert_progressif"

    if (
        profil_prog in ["progression_rapide", "progression_lente"]
        and profil_modif in ["petit_ajusteur", "modificateur_modere"]
        and profil_tent in ["resolution_rapide", "resolution_moyenne"]
        and profil_temps in ["rapide", "modere"]
    ):
        return "reviseur_methodique"

    if (
        profil_prog in ["progression_rapide", "progression_lente"]
        and profil_modif == "gros_restructurateur"
    ):
        return "gros_restructurateur"

    if (
        profil_prog in ["stagnation", "regression"]
        and profil_tent == "resolution_longue"
        and profil_temps in ["modere", "lent"]
    ):
        return "bloque"

    if (
        profil_prog in ["stagnation", "regression"]
        and profil_modif == "gros_restructurateur"
    ):
        return "explorateur_chaotique"

    if profil_temps == "lent" and profil_prog == "progression_lente":
        return "lent_mais_regulier"

    return "profil_intermediaire"

def pre_calcul(dfo, solution_df):
    """
    Construit un dataset allégé pour la classification des users.

    Colonnes conservées :
    - id
    - exercice
    - type_exercice
    - t
    - t_plus_1
    - distance_zss
    - score_t_solution
    - score_t_plus_1_solution
    - progression_solution
    - temps_t
    - temps_t_plus_1
    - delta_temps
    """
    if "code" not in dfo.columns:
        raise ValueError(f"Colonne 'code' absente. Colonnes disponibles : {dfo.columns.tolist()}")

    if "temps_passe" not in dfo.columns:
        raise ValueError(f"Colonne 'temps_passe' absente. Colonnes disponibles : {dfo.columns.tolist()}")

    df = dfo[dfo["code"].notna() & (dfo["code"] != "")].copy()

    couples = couples_valides(dfo)
    df = df[df[["id_compte", "level_1"]].apply(tuple, axis=1).isin(couples)].copy()
    solution_dict = build_solution_dict(solution_df)

    lignes = []

    user_groups = list(df.groupby("id_compte", sort=False))

    for user_id, user_df in tqdm(
        user_groups,
        desc="Processing",
        position=0,
        leave=True,
        dynamic_ncols=True
    ):
        ex_groups = list(user_df.groupby("level_1", sort=False))

        for ex, group in tqdm(
            ex_groups,
            desc=f"User {user_id}",
            position=1,
            leave=False
        ):
            group = group.reset_index(drop=True)

            if len(group) < 2:
                continue

            sol_info = solution_dict.get(ex)

            if sol_info is not None:
                correct_codes = sol_info.get("correctCodes", [])
                exercise_type = sol_info.get("exerciseType", np.nan)

                ast_solutions = []
                for sol in correct_codes:
                    ast_sol = code_to_ast(sol)
                    if ast_sol is not None:
                        ast_solutions.append(ast_sol)
            else:
                exercise_type = np.nan
                ast_solutions = []

            zss_cache = {}

            for i in tqdm(
                range(len(group) - 1),
                desc=f"Exercice {ex}",
                position=2,
                leave=False
            ):
                code_t = group.loc[i, "code"]
                code_t_1 = group.loc[i + 1, "code"]

                # 0 = temps non renseigné
                temps_t = group.loc[i, "temps_passe"]
                temps_t_1 = group.loc[i + 1, "temps_passe"]
                temps_t = np.nan if pd.isna(temps_t) or temps_t == 0 else temps_t
                temps_t_1 = np.nan if pd.isna(temps_t_1) or temps_t_1 == 0 else temps_t_1

                if pd.notna(temps_t) and pd.notna(temps_t_1):
                    delta_temps = temps_t_1 - temps_t
                else:
                    delta_temps = np.nan

                try:
                    tree_t = get_zss_tree(code_t, zss_cache)
                    tree_t_1 = get_zss_tree(code_t_1, zss_cache)

                    if tree_t is not None and tree_t_1 is not None:
                        distance_zss, _ = distance(tree_t, tree_t_1, get_children)
                    else:
                        distance_zss = np.nan
                except Exception as err:
                    tqdm.write(f"Erreur distance user={user_id}, ex={ex}, t={i+1}: {err}")
                    distance_zss = np.nan

                ast_t = code_to_ast(code_t)
                ast_t_1 = code_to_ast(code_t_1)

                try:
                    if ast_t is not None and ast_solutions:
                        comp_t_sol = prog_vs_answer(ast_t, ast_solutions)
                    else:
                        comp_t_sol = [0, {}]
                except Exception as err:
                    tqdm.write(f"Erreur comparaison t->solution user={user_id}, ex={ex}, t={i+1}: {err}")
                    comp_t_sol = [0, {}]

                try:
                    if ast_t_1 is not None and ast_solutions:
                        comp_t1_sol = prog_vs_answer(ast_t_1, ast_solutions)
                    else:
                        comp_t1_sol = [0, {}]
                except Exception as err:
                    tqdm.write(f"Erreur comparaison t+1->solution user={user_id}, ex={ex}, t={i+2}: {err}")
                    comp_t1_sol = [0, {}]

                score_t_sol = extract_score(comp_t_sol)
                score_t1_sol = extract_score(comp_t1_sol)
                progression_sol = score_t1_sol - score_t_sol

                lignes.append({
                    "id": user_id,
                    "exercice": ex,
                    "type_exercice": exercise_type,
                    "t": i + 1,
                    "t_plus_1": i + 2,
                    "distance_zss": distance_zss,
                    "score_t_solution": score_t_sol,
                    "score_t_plus_1_solution": score_t1_sol,
                    "progression_solution": progression_sol,
                    "temps_t": temps_t,
                    "temps_t_plus_1": temps_t_1,
                    "delta_temps": delta_temps
                })

    dataset = pd.DataFrame(lignes)

    save_dataset_to_json(dataset, "pre_calcul.json")

    return dataset

def build_user_classification(dataset):
    df = dataset.copy()

    user_stats = (
        df.groupby("id", dropna=False)
        .agg(
            mean_progress=("progression_solution", "mean"),
            mean_zss=("distance_zss", "mean"),
            max_tentative=("t_plus_1", "max"),
            nb_transitions=("id", "size"),
            mean_delta_temps=("delta_temps", "mean"),
            nb_temps_valides=("delta_temps", lambda s: s.notna().sum())
        )
        .reset_index()
    )

    user_stats["profil_progression"] = user_stats["mean_progress"].apply(profil_progression)
    user_stats["profil_modification"] = user_stats["mean_zss"].apply(profil_modification)
    user_stats["profil_tentatives"] = user_stats["max_tentative"].apply(profil_tentatives)

    user_stats["profil_temps"] = user_stats.apply(
        lambda row: "non_renseigne"
        if row["nb_temps_valides"] == 0
        else profil_temps(row["mean_delta_temps"]),
        axis=1
    )

    user_stats["classe"] = user_stats.apply(
        lambda row: classe_finale_from_profils(
            row["profil_progression"],
            row["profil_modification"],
            row["profil_temps"],
            row["profil_tentatives"]
        ),
        axis=1
    )

    save_dataset_to_json(dataset, "classif_user.json")

    return user_stats

def build_user_exercice_classification(dataset):
    """
    Classification par couple (user, exercice) avec profils détaillés.

    Colonnes attendues dans dataset :
    - id
    - exercice
    - type_exercice
    - progression_solution
    - distance_zss
    - t_plus_1
    - delta_temps

    Retour
    ------
    pandas.DataFrame
        Colonnes :
        - id
        - exercice
        - type_exercice
        - mean_progress
        - mean_zss
        - max_tentative
        - nb_transitions
        - mean_delta_temps
        - nb_temps_valides
        - profil_progression
        - profil_modification
        - profil_tentatives
        - profil_temps
        - classe
    """
    df = dataset.copy()

    stats = (
        df.groupby(["id", "exercice"], dropna=False)
        .agg(
            type_exercice=("type_exercice", "first"),
            mean_progress=("progression_solution", "mean"),
            mean_zss=("distance_zss", "mean"),
            max_tentative=("t_plus_1", "max"),
            nb_transitions=("id", "size"),
            mean_delta_temps=("delta_temps", "mean"),
            nb_temps_valides=("delta_temps", lambda s: s.notna().sum())
        )
        .reset_index()
    )

    stats["profil_progression"] = stats["mean_progress"].apply(profil_progression)
    stats["profil_modification"] = stats["mean_zss"].apply(profil_modification)
    stats["profil_tentatives"] = stats["max_tentative"].apply(profil_tentatives)

    stats["profil_temps"] = stats.apply(
        lambda row: "non_renseigne"
        if row["nb_temps_valides"] == 0
        else profil_temps(row["mean_delta_temps"]),
        axis=1
    )

    stats["classe"] = stats.apply(
        lambda row: classe_finale_from_profils(
            row["profil_progression"],
            row["profil_modification"],
            row["profil_temps"],
            row["profil_tentatives"]
        ),
        axis=1
    )

    save_dataset_to_json(dataset, "classif_user_ex.json")

    return stats