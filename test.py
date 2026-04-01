import utils
import numpy as np
from pandasgui import show

# Lecture des donnees

# lit le fichier brut
# data = utils.read_data("data/2025.json")
# sol = utils.read_data("data/exercises.json")

# transforme les donnees AlgoPython en DataFrame exploitable
# df = utils.AlgoPython_data(data)

# Tests

# analyse detaillee d'un user
# utils.analyse_user(46272, df)

# comparaison manuelle entre deux programmes si besoin
# p1 = utils.code_to_ast(df.iloc[0]["code"])
# p2 = utils.code_to_ast(df.iloc[1]["code"])
# print(utils.ast_dump(p1))
# print(utils.primary_code_error_two_prog(p1, p2))

# affichage du DataFrame si besoin
# show(df[df["code"] != ""])

# Echantillonnage
# c = utils.cas1(df, sol)
# c = utils.read_data("data/cas1_2022.json")
# show(c)

# c2 = utils.cas2(c)
c2 = utils.read_data("data/cas2_2022.json")
# show(c2)

id = np.random.choice(c2["id"], 1)[0]
ex = np.random.choice(c2[c2["id"] == id]["exercice"], 1)[0]
cas = c2[(c2["id"] == id) & (c2["exercice"] == ex)].reset_index(drop=True)

print(f"Information sur les tentatives de l'user {id} sur l'exercice {ex}")
for i in range(len(cas)):
    primary_list = cas.loc[i, "primary_code_errors_text"]
    typology_list = cas.loc[i, "typology_based_code_error_text"]
    print(f"Tentative {i+1} vs {i+2}")
    print(f"Code {i+1}:")
    print(cas.loc[i, "code_t"])
    print()
    print(f"Code {i+2}:")
    print(cas.loc[i, "code_t_1"])
    print()
    if len(typology_list) > 0:
        print("Modification simple")
        for k in range(len(typology_list)):
            print(typology_list[k])
        print()
    if len(primary_list) > 0:
        print("Modification détaillé")
        for j in range(len(primary_list)):
            print(primary_list[j])
    if len(primary_list) == 0 and  len(typology_list) == 0:
        print("Pas de modification")
    print("______________________________")
