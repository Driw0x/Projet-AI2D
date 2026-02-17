import utils
from pandasgui import show
import numpy as np

data = utils.read_data("data/2025.json")
df = utils.AlgoPython_data(data)

# show(df)

# print(utils.code_to_ast(df["code"][6]))
# print(utils.ast_dump(utils.code_to_ast(df["code"][6])))

# # Caracteristique des donnees
# print("Nb de donnees: ", df.shape[0])
# print("Nb d'etablissement: ", len(np.unique(df["nom_etab"])))
# print("Nb d'utilisateur: ", len(np.unique(df["display_name"])))


