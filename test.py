import utils
from pandasgui import show

data = utils.read_data("data/2022-08-16_to_2023-08-15-trajectories-slice-filtered-schools.json")
df = utils.AlgoPython_data(data)

# show(df)

print(utils.code_to_ast(df["code"][6]))
print(utils.ast_dump(utils.code_to_ast(df["code"][6])))


