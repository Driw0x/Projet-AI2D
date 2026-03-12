import pandas as pd
import ast_error_detection as aed
import ast
import numpy as np

def code_to_ast(code):
    return ast.parse(code)

def ast_dump(t):
    return ast.dump(t, indent=2)
 
def read_data(path):
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

def AlgoPython_data(df):

    classes = df[df.columns]
    classes = classes.explode("classes", ignore_index=True)

    classes_details = pd.json_normalize(classes["classes"])
    classes = classes.drop(columns=["classes"])
    classes = classes.join(classes_details)

    comptes = classes.explode("comptes", ignore_index=True)

    comptes_details = pd.json_normalize(comptes["comptes"], max_level=0)
    comptes = comptes.drop(columns=["comptes"])
    comptes = comptes.join(comptes_details)

    trajectories = comptes["trajectories"]
    trajectories = trajectories.where(trajectories.map(type).eq(dict), {})

    traj_wide = trajectories.apply(pd.Series)

    traj_long = traj_wide.stack(future_stack=True).reset_index(name="tentatives")

    contexte = comptes.drop(columns=["trajectories"])
    traj_long = traj_long.join(contexte, on="level_0")
    traj_long = traj_long.drop(columns=["level_0"])

    traj_long = traj_long.explode("tentatives", ignore_index=True)

    tentatives = pd.json_normalize(traj_long["tentatives"])

    df_final = traj_long.drop(columns=["tentatives"]).join(tentatives)

    df_final = df_final[df_final["code"].notna()]

    df_final = df_final.reset_index(drop=True)
    df_final = df_final[(df_final["statut"] != "err") & (df_final["statut"] != "ask")]
    # Transformation des OK en 1 et KO en 0 pour calculer les caracteristiques statistiques
    # df_final["statut"] = np.where(df_final["statut"] == "ok", 1, 0)

    return df_final

def primary_code_error_two_prog(p1, p2):
    return aed.get_primary_code_errors(p1, p2)

def prog_vs_answer(p1, list_answer):
    return aed.get_typology_based_code_error(p1, list_answer)

def regle(primary_code_error):
    def extract_function_name(s):
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
    prog = []
    p = df.reset_index(drop=True)
    for i in range(len(p)):
        prog.append(p.iloc[i]["code"])
    
    answer = p.iloc[len(p) - 1]["code"]
    ast_answer = code_to_ast(answer)

    for i in range(len(prog)-1):
        print(f"Tentative {i + 1}: ")
        if prog[i]:
            ast_p = code_to_ast(prog[i])
            regle(primary_code_error_two_prog(ast_p, ast_answer))
        else:
            print("Code vide")

def analyse_user(id_compte, df):
    u = df[(df["id_compte"] == id_compte)]
    ex = np.unique(u["level_1"])
    for e in ex:
        print(f"Exercice {e}:")
        p = u[u["level_1"] == e]
        if len(p) > 1:
            comparaison_tentative_solution(p)
        else:
            print("Qu'un seul essaie")
            if p["statut"].iloc[0] == 1:
                print("Il a réussi du premier coup")
            else:
                print("Il a abandonné dès le premier essaie")
        print()

def recherche_echantillon(df):
    df = df[df["code"] != ""]
    d = {}
    for id in np.unique(df["id_compte"]):
        print(f"User {id}")
        for ex in np.unique(df["level_1"]):
            if len(df[(df["id_compte"] == id) & (df["level_1"] == ex)]) > 5:
                if ex not in d:
                    d[ex] = 0
                d[ex] += 1
                print(f"Exercice {ex}")
        print()
    print(d)