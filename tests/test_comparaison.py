import numpy as np

import tools.comparaison as comparaison


def test_primary_code_error_two_prog_returns_empty_result_on_aed_error(monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("AED indisponible")

    monkeypatch.setattr(comparaison.aed, "get_primary_code_errors", raise_error)

    assert comparaison.primary_code_error_two_prog("x=1", "x=2") == [0, []]


def test_prog_vs_answer_returns_empty_result_on_aed_error(monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("AED indisponible")

    monkeypatch.setattr(comparaison.aed, "get_typology_based_code_error", raise_error)

    assert comparaison.prog_vs_answer("x=1", ["x=2"]) == [0, {}]


def test_extract_score_accepts_valid_comparison_result():
    assert comparaison.extract_score([0.75, []]) == 0.75
    assert np.isnan(comparaison.extract_score({"bad": "format"}))


def test_compare_transition_returns_expected_keys(monkeypatch):
    monkeypatch.setattr(
        comparaison,
        "primary_code_error_two_prog",
        lambda code1, code2: [0, []],
    )
    monkeypatch.setattr(
        comparaison,
        "prog_vs_answer",
        lambda code1, answers: [0.2, []],
    )

    result = comparaison.compare_transition(
        "x = 1",
        "x = 2",
        include_code_errors=True,
        include_solution_scores=True,
        ast_solutions=[comparaison.code_to_ast("x = 2")],
    )

    assert {
        "distance_zss",
        "ops",
        "primary_code_errors_score",
        "primary_code_errors",
        "typology_based_code_error_score",
        "typology_based_code_error",
        "score_t_solution",
        "score_t_plus_1_solution",
        "progression_solution",
    }.issubset(result.keys())
    assert result["progression_solution"] == 0


def test_comparaison_sets_tags_from_aed_results(monkeypatch):
    monkeypatch.setattr(
        comparaison.aed,
        "get_typology_based_code_error",
        lambda code1, answers: [0, ["LO_FOR_MISSING", "LO_FOR_NUMBER_ITERATION_ERROR"]],
    )
    monkeypatch.setattr(
        comparaison,
        "primary_code_error_two_prog",
        lambda code1, code2: [0, [["MISSING_IF_STATEMENT", "if", "Module > Call[0]"]]],
    )

    tag = comparaison.comparaison("x=1", "for i in range(3):\n    x=1")

    assert tag.ajout_boucle_for is True
    assert tag.modif_boucle_for_iteration is True
    assert tag.ajout_if is True
    assert tag.modif_hors_struct is True


def test_evo_code_tracks_first_for_and_modification_range(monkeypatch):
    class FakeTag:
        ajout_boucle_for = True
        modif_boucle_for_iteration = False
        modif_corps_boucle_for = False
        ajout_boucle_while = False
        modif_boucle_while_iteration = False
        modif_corps_boucle_while = False
        ajout_if = False
        modif_cond_if = False
        modif_corps_if = False
        modif_hors_struct = False

    monkeypatch.setattr(comparaison, "comparaison", lambda code1, code2: FakeTag())

    evo = comparaison.evo_code(["x=1", "x=2"])

    assert evo.premiere_boucle_for == 1
    assert evo.range_modif_for == [[0, 2]]
