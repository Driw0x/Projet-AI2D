import pandas as pd

import tools.classification as classification


def sample_precalc():
    return pd.DataFrame({
        "id": [1, 1, 2, 2],
        "exercice": ["A1", "A1", "B1", "B1"],
        "type_exercice": ["design", "design", "design", "design"],
        "t_plus_1": [2, 3, 2, 3],
        "progression_solution": [0.2, 0.4, -0.1, 0.0],
        "distance_zss": [1, 3, 10, 12],
        "delta_temps": [10, 20, 100, 120],
        "reussite_finale_exercice": [1, 1, 0, 0],
    })


def test_profile_helpers():
    thresholds = {"progress": (0.1, 0.3), "zss": (2, 5), "tentatives": (3, 5), "temps": (30, 90)}
    assert classification.profil_progression_dynamic(-1, thresholds) == "regression"
    assert classification.profil_modification_dynamic(1, thresholds) == "petit_ajusteur"
    assert classification.profil_tentatives_dynamic(6, thresholds) == "resolution_longue"
    assert classification.profil_temps_dynamic(100, 1, thresholds) == "lent"
    assert classification.profil_reussite_exercice(1) == "reussi"
    assert classification.profil_taux_reussite(0.8) == "forte_reussite"


def test_classification_pipeline():
    df = sample_precalc()
    stats = classification.aggregate_classification_stats(df, ["id"], "user")
    profiled, thresholds = classification.apply_dynamic_profiles(stats)
    assert "classe" in profiled.columns
    assert set(thresholds.keys()) == {"progress", "zss", "tentatives", "temps"}
    assert len(classification.build_user_classification(df)[0]) == 2
    assert len(classification.build_user_exercice_classification(df)[0]) == 2


def test_distribution_and_final_class():
    dist = classification.distribution_classes(pd.DataFrame({"classe": ["a", "a", "b"]}))
    assert set(dist.columns) == {"classe", "nb", "%"}
    assert classification.classe_finale_from_profils(
        "progression_rapide",
        "petit_ajusteur",
        "rapide",
        "resolution_rapide",
        "reussi",
    ) == "expert_progressif"
