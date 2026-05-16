import numpy as np
import pandas as pd
import pytest

import tools.dataset_builder as dataset_builder


def fake_compare_transition(*args, **kwargs):
    return {
        "distance_zss": 1,
        "ops": {},
        "primary_code_errors_score": 0,
        "primary_code_errors": [],
        "typology_based_code_error_score": 0,
        "typology_based_code_error": {},
        "score_t_solution": 0.2,
        "score_t_plus_1_solution": 0.5,
        "progression_solution": 0.3,
    }


def sample_attempts():
    return pd.DataFrame({
        "id_compte": [1, 1, 1, 1, 1, 1],
        "level_1": ["A1"] * 6,
        "code": ["x=1", "x=2", "x=3", "x=4", "x=5", "x=6"],
        "temps_passe": [1, 2, 0, 4, 5, 6],
        "statut": [0, 0, 0, 0, 0, 1],
    })


def sample_solutions():
    return pd.DataFrame({
        "exerciseTitle": ["A-1 Test"],
        "correctCodes": [["x=6"]],
        "exerciseType": ["design"],
    })


def test_helpers():
    with pytest.raises(ValueError):
        dataset_builder.check_required_columns(pd.DataFrame({"a": [1]}), ["a", "b"])
    assert dataset_builder.build_solution_dict(sample_solutions())["A1"]["exerciseType"] == "design"
    assert dataset_builder.build_user_exercise_count_dict(sample_attempts())[(1, "A1")] == 6
    assert dataset_builder.couples_valides(sample_attempts(), min_tentatives=5) == {(1, "A1")}
    assert len(dataset_builder.prepare_attempts_dataframe(sample_attempts(), min_tentatives=5)) == 6
    assert len(dataset_builder.build_ast_solutions({"correctCodes": ["x=1", "for"]})) == 1
    assert np.isnan(dataset_builder.safe_temps(0))


def test_build_transition_dataset(monkeypatch):
    monkeypatch.setattr(dataset_builder, "compare_transition", fake_compare_transition)
    out = dataset_builder.build_transition_dataset(
        sample_attempts(),
        sample_solutions(),
        min_tentatives=5,
    )
    assert len(out) == 5
    assert {"id", "exercice", "distance_zss", "progression_solution"}.issubset(out.columns)
