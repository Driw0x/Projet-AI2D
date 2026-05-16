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
        "typology_based_code_error": [{}],
    })


def test_cas1_x_y_and_cas1(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(cas, "compare_transition", fake_compare_transition)
    monkeypatch.setattr(cas, "build_transition_dataset", fake_build_transition_dataset)

    df = pd.DataFrame({"code": ["x=1", "x=2"]})
    assert cas.cas1_x_y(df, 0, 1)[0] == 2

    out = cas.cas1(pd.DataFrame(), pd.DataFrame())
    assert "dist_zss" in out.columns


def test_regle_and_cas2(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    phrases, unknown = cas.regle(
        primary_code_errors=[
            ["MISSING_IF_STATEMENT", "if", "Module > If[0]"]
        ]
    )
    assert phrases
    assert unknown == set()

    df = pd.DataFrame({
        "primary_code_errors": [[["MISSING_IF_STATEMENT", "if", "Module > If[0]"]]],
        "typology_based_code_error": [["LO_FOR_MISSING"]],
    })
    out = cas.cas2(df)
    assert "primary_code_errors_text" in out.columns
    assert "typology_based_code_error_text" in out.columns
