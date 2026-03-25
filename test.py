import utils
from pandasgui import show
import numpy as np

# Lecture des donnees

# lit le fichier brut
data = utils.read_data("data/2025.json")
sol = utils.read_data("data/exercises.json")

# transforme les donnees AlgoPython en DataFrame exploitable
df = utils.AlgoPython_data(data)

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

# utils.build_datasets(df, sol)