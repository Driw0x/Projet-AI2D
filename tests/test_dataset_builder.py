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
        "id_compte": [1, 1, 1, 1, 1, 1, 2],
        "level_1": ["A1", "A1", "A1", "A1", "A1", "A1", "B1"],
        "code": ["x=1", "x=2", "x=3", "x=4", "x=5", "x=6", "y=1"],
        "temps_passe": [1, 2, 0, 4, 5, 6, 7],
        "statut": [0, 0, 0, 0, 0, 1, 0],
    })


def sample_solutions():
    return pd.DataFrame({
        "exerciseTitle": ["A-1 Test"],
        "correctCodes": [["x=6"]],
        "exerciseType": ["design"],
    })


def test_check_required_columns_raises_when_column_is_missing():
    with pytest.raises(ValueError):
        dataset_builder.check_required_columns(pd.DataFrame({"a": [1]}), ["a", "b"])


def test_build_solution_dict_normalises_exercise_title():
    solution_dict = dataset_builder.build_solution_dict(sample_solutions())

    assert solution_dict["A1"]["exerciseType"] == "design"
    assert solution_dict["A1"]["correctCodes"] == ["x=6"]


def test_user_exercise_helpers_keep_only_couples_with_enough_attempts():
    attempts = sample_attempts()

    assert dataset_builder.build_user_exercise_count_dict(attempts)[(1, "A1")] == 6
    assert dataset_builder.couples_valides(attempts, min_tentatives=5) == {(1, "A1")}
    assert len(dataset_builder.prepare_attempts_dataframe(attempts, min_tentatives=5)) == 6


def test_build_ast_solutions_ignores_invalid_python_code():
    trees = dataset_builder.build_ast_solutions({"correctCodes": ["x=1", "for"]})

    assert len(trees) == 1


def test_safe_temps_converts_zero_to_nan():
    assert dataset_builder.safe_temps(10) == 10
    assert np.isnan(dataset_builder.safe_temps(0))


def test_build_transition_dataset_creates_one_row_per_transition(monkeypatch):
    monkeypatch.setattr(dataset_builder, "compare_transition", fake_compare_transition)

    out = dataset_builder.build_transition_dataset(
        sample_attempts(),
        sample_solutions(),
        min_tentatives=5,
    )

    assert len(out) == 5
    assert {
        "id",
        "exercice",
        "type_exercice",
        "t",
        "t_plus_1",
        "distance_zss",
        "progression_solution",
        "delta_temps",
        "reussite_finale_exercice",
    }.issubset(out.columns)
