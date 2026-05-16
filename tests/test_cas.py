import pandas as pd

import tools.cas as cas


def fake_compare_transition(*args, **kwargs):
    return {
        "distance_zss": 2,
        "ops": {},
        "primary_code_errors_score": 0,
        "primary_code_errors": [],
        "typology_based_code_error_score": 0,
        "typology_based_code_error": {},
    }


def fake_build_transition_dataset(*args, **kwargs):
    return pd.DataFrame({
        "id": [1],
        "exercice": ["A1"],
        "type_exercice": ["design"],
        "t": [1],
        "t_plus_1": [2],
        "distance_zss": [1],
        "ops": [{}],
        "code_t": ["x=1"],
        "code_t_1": ["x=2"],
        "primary_code_errors_score": [0],
        "primary_code_errors": [[]],
        "typology_based_code_error_score": [0],
        "typology_based_code_error": [[]],
    })


def test_cas1_x_y_returns_nan_for_invalid_indices():
    df = pd.DataFrame({"code": ["x=1"]})

    distance, ops, primary, typology = cas.cas1_x_y(df, 0, 5)

    assert pd.isna(distance)
    assert ops == {}
    assert primary == (0, [])
    assert typology == [0, {}]


def test_cas1_x_y_compares_two_rows(monkeypatch):
    monkeypatch.setattr(cas, "compare_transition", fake_compare_transition)
    df = pd.DataFrame({"code": ["x=1", "x=2"]})

    distance, ops, primary, typology = cas.cas1_x_y(df, 0, 1)

    assert distance == 2
    assert ops == {}
    assert primary == (0, [])
    assert typology == [0, {}]


def test_cas1_builds_dataset_and_saves_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(cas, "build_transition_dataset", fake_build_transition_dataset)

    out = cas.cas1(pd.DataFrame(), pd.DataFrame())

    assert "dist_zss" in out.columns
    assert (tmp_path / "data" / "cas1.json").exists()


def test_regle_translates_known_primary_error():
    phrases, unknown = cas.regle(
        primary_code_errors=[["MISSING_IF_STATEMENT", "if", "Module > If[0]"]]
    )

    assert phrases
    assert unknown == set()


def test_cas2_adds_text_columns(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = pd.DataFrame({
        "primary_code_errors": [[["MISSING_IF_STATEMENT", "if", "Module > If[0]"]]],
        "typology_based_code_error": [["LO_FOR_MISSING"]],
    })

    out = cas.cas2(df)

    assert "primary_code_errors_text" in out.columns
    assert "typology_based_code_error_text" in out.columns
    assert (tmp_path / "data" / "cas2.json").exists()
