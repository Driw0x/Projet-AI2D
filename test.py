import utils
from pandasgui import show
import numpy as np

data = utils.read_data("data/2025.json")
df = utils.AlgoPython_data(data)

# show(df[df["code"] != ""])

p = df[(df["level_1"] == "B18") & (df["id_classe"] == 2234)]

p1 = utils.code_to_ast(df.iloc[0]["code"])
p2 = utils.code_to_ast(df.iloc[1]["code"])
p3 = utils.code_to_ast(df.iloc[2]["code"])
p4 = utils.code_to_ast(df.iloc[3]["code"])
# print(p1)
# print(utils.ast_dump(p1))
print(utils.prog_vs_answer(p3, [p4]))

# # Caracteristique des donnees
# print(df.columns)
# print(df.shape)
# print(df.shape[0], "donnees")
# print(len(np.unique(df["nom_etab"])), "etablissements")
# print(len(np.unique(df["display_name"])), "utilisateurs")
# print("Moyenne statut OK: ", np.mean(df["statut"]))
# print("Ecart-type statut OK: ", np.std(df["statut"]))
# print("Moyenne nb tentative: ", np.mean(df["nb_tentative"]))
# print("Ecart-type nb tentative: ", np.std(df["nb_tentative"]))
# print("Moyenne temps passe: ", np.mean(df["temps_passe"]))
# print("Ecart-type temps passe: ", np.std(df["temps_passe"]))

utils.recherche_echantillon(df)
