from tools.comparaison import compare_transition
from tools.dataset_builder import build_transition_dataset
import numpy as np
from tools.io_tools import save_dataset_to_json
from tqdm import trange

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