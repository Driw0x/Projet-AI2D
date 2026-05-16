import ast_error_detection as aed
from tqdm import tqdm
import numpy as np
from tools.ast_tools import *
from ast_error_detection.zang_shasha_distance import distance
import warnings
from models.differences_tag import Differences_tag
from models.evolution_code import Evolution_code

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


# Obtenir les différents tags nécessaires entre 2 codes
def comparaison(code1, code2):
    modif_base = aed.get_typology_based_code_error(code1, [code2])[1]
    modif_detaillee = primary_code_error_two_prog(code1, code2)[1]
    
    tag = Differences_tag()

    for differences in modif_base:
        if differences.startswith("EXP_MISSING_ASSIGNMENT_MISSING"):
            tag.ajout_variable = True

        elif differences.startswith("LO_FOR_MISSING"):
            tag.ajout_boucle_for = True
        elif differences.startswith("LO_FOR_NUMBER_ITERATION_ERROR"):
            tag.modif_boucle_for_iteration = True

        elif differences.startswith("LO_WHILE_MISSING"):
            tag.ajout_boucle_while = True
        elif differences.startswith("LO_WHILE_NUMBER_ITERATION"):
            tag.modif_boucle_while_iteration = True

    for i in range(len(modif_detaillee)):
        if modif_detaillee[i][2].startswith("Module > For"):
            tag.modif_corps_boucle_for = True
        if modif_detaillee[i][2].startswith("Module > While"):
            tag.modif_corps_boucle_while = True
        if modif_detaillee[i][0].startswith("MISSING_IF_STATEMENT"):
            tag.ajout_if = True
        if modif_detaillee[i][2].startswith("Module > Call"):
            tag.modif_hors_struct = True

    return tag


# Obtenir toutes les modifs à analyser sur une série de code
def evo_code(codes):
    # Ajout du code vide au début
    codes = [""] + codes

    # Fonction pour mettre à jour les range de modifications
    def update_range(bool, liste_range, i):
        if bool:
            if len(liste_range) == 0:
                liste_range.append([i])
            elif len(liste_range[-1]) == 2:
                liste_range.append([i])
        else:
            if len(liste_range) != 0 and len(liste_range[-1]) == 1:
                liste_range[-1].append(i)

    if len(codes) <= 1:
        return
    
    # La structure à retourner
    evo = Evolution_code()
    
    # On compare 2 codes successifs
    for i in range(len(codes) - 1):
        code1 = codes[i]
        code2 = codes[i+1]

        tag = comparaison(code1, code2)

        # On gère les apparitions des structures de contrôle
        if evo.premiere_boucle_for == -1 and tag.ajout_boucle_for:
            evo.premiere_boucle_for = i+1
        if evo.premiere_boucle_while == -1 and tag.ajout_boucle_while:
            evo.premiere_boucle_while = i+1
        if evo.premiere_struct_if == -1 and tag.ajout_if:
            evo.premiere_struct_if = i+1

        # On met à jour les ranges
        update_range(tag.ajout_boucle_for or tag.modif_boucle_for_iteration or tag.modif_corps_boucle_for, evo.range_modif_for, i)
        update_range(tag.ajout_boucle_while or tag.modif_boucle_while_iteration or tag.modif_corps_boucle_while, evo.range_modif_while, i)
        update_range(tag.ajout_if or tag.modif_cond_if or tag.modif_corps_if, evo.range_modif_if, i)
        update_range(tag.modif_hors_struct, evo.range_modif_hors_struct, i)

    
    # Fin des modif sur les structures (ou hors struct) s'il y en avait des incomplètes
    if len(evo.range_modif_for) != 0 and len(evo.range_modif_for[-1]) == 1:
        evo.range_modif_for[-1].append(len(codes)-1)
    if len(evo.range_modif_while) != 0 and len(evo.range_modif_while[-1]) == 1:
        evo.range_modif_while[-1].append(len(codes)-1)
    if len(evo.range_modif_if) != 0 and len(evo.range_modif_if[-1]) == 1:
        evo.range_modif_if[-1].append(len(codes)-1)
    if len(evo.range_modif_hors_struct) != 0 and len(evo.range_modif_hors_struct[-1]) == 1:
        evo.range_modif_hors_struct[-1].append(len(codes)-1)

    return evo
