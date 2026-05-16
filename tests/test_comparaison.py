import tools.comparaison as comparaison


def test_comparaison_sets_tags(monkeypatch):
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


def test_evo_code_tracks_first_for_and_ranges(monkeypatch):
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
