import utils
from pandasgui import show

df = utils.read_data("data/2022-08-16_to_2023-08-15-trajectories-slice-filtered-schools.json")

show(utils.AlgoPython_data(df))