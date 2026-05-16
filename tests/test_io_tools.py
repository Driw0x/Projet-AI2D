import json

import numpy as np
import pandas as pd

from tools.io_tools import (
    AlgoPython_data,
    clean_value,
    explode_json_column,
    read_data,
    save_dataset_to_json,
    trajectories_to_long,
)


def test_read_data_supports_json_and_csv(tmp_path):
    json_path = tmp_path / "data.json"
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2]}).to_json(json_path)
    pd.DataFrame({"a": [1, 2]}).to_csv(csv_path, index=False)

    assert list(read_data(str(json_path))["a"]) == [1, 2]
    assert list(read_data(str(csv_path))["a"]) == [1, 2]


def test_explode_json_column_turns_list_of_dicts_into_columns():
    df = pd.DataFrame({"items": [[{"x": 1}, {"x": 2}]]})

    out = explode_json_column(df, "items")

    assert list(out["x"]) == [1, 2]


def test_trajectories_to_long_creates_one_row_per_exercise():
    comptes = pd.DataFrame({
        "id_compte": [1],
        "trajectories": [{"A1": [{"code": "x=1"}], "B1": [{"code": "y=1"}]}],
    })

    out = trajectories_to_long(comptes)

    assert set(out["level_1"]) == {"A1", "B1"}
    assert isinstance(out.loc[0, "tentatives"], list)


def test_algopython_data_flattens_minimal_nested_json():
    raw = pd.DataFrame({
        "classes": [[{
            "nom": "classe1",
            "comptes": [{
                "id_compte": 1,
                "trajectories": {
                    "A1": [
                        {"code": "x=1", "statut": "ok", "temps_passe": 5},
                        {"code": "x=2", "statut": "nok", "temps_passe": 8},
                    ]
                },
            }],
        }]],
    })

    out = AlgoPython_data(raw)

    assert len(out) == 2
    assert list(out["statut"]) == [1, 0]
    assert out.loc[0, "level_1"] == "A1"


def test_clean_value_converts_numpy_values_to_json_safe_values():
    cleaned = clean_value({"a": np.int64(1), "b": np.float64(np.nan), "c": {2, 1}})

    assert cleaned["a"] == 1
    assert cleaned["b"] is None
    assert sorted(cleaned["c"]) == [1, 2]


def test_save_dataset_to_json_writes_into_data_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    save_dataset_to_json(pd.DataFrame({"a": [1], "b": [np.nan]}), "out.json")

    with open(tmp_path / "data" / "out.json", encoding="utf-8") as f:
        assert json.load(f) == [{"a": 1, "b": None}]
