import pandas as pd
import json

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

import pandas as pd

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

    return df_final.reset_index(drop=True)

