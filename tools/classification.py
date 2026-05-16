import pandas as pd
from tools.dataset_builder import check_required_columns, build_transition_dataset

def remove_outliers_iqr(series):
    """
    Retire les valeurs extrêmes avec la règle IQR.
    """
    s = series.dropna()
    if len(s) == 0:
        return s

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return s

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = s[(s >= lower) & (s <= upper)]
    return filtered if len(filtered) > 0 else s


def compute_tertile_thresholds(series, fallback=(0.33, 0.66)):
    """
    Calcule les seuils tertiles après retrait des outliers.
    """
    s = remove_outliers_iqr(series)

    if len(s) == 0:
        return fallback

    t1 = s.quantile(1 / 3)
    t2 = s.quantile(2 / 3)
    return t1, t2


def compute_dynamic_thresholds(user_stats):
    """
    Calcule tous les seuils dynamiques à partir des stats agrégées.
    """
    thresholds = {}

    thresholds["progress"] = compute_tertile_thresholds(
        user_stats["mean_progress"],
        fallback=(0.0, 1.0)
    )

    thresholds["zss"] = compute_tertile_thresholds(
        user_stats["mean_zss"],
        fallback=(3.0, 8.0)
    )

    thresholds["tentatives"] = compute_tertile_thresholds(
        user_stats["max_tentative"],
        fallback=(3, 6)
    )

    thresholds["temps"] = compute_tertile_thresholds(
        user_stats["mean_delta_temps"],
        fallback=(30, 120)
    )

    return thresholds


def profil_progression_dynamic(mean_progress, thresholds):
    """
    Profil de progression avec seuils dynamiques.
    """
    if pd.isna(mean_progress):
        return "inconnu"

    t1, t2 = thresholds["progress"]

    if mean_progress < 0:
        return "regression"
    if mean_progress <= t1:
        return "stagnation"
    if mean_progress <= t2:
        return "progression_lente"
    return "progression_rapide"


def profil_modification_dynamic(mean_zss, thresholds):
    """
    Profil de modification avec seuils dynamiques.
    """
    if pd.isna(mean_zss):
        return "inconnu"

    t1, t2 = thresholds["zss"]

    if mean_zss <= t1:
        return "petit_ajusteur"
    if mean_zss <= t2:
        return "modificateur_modere"
    return "gros_restructurateur"


def profil_tentatives_dynamic(nb_tentatives, thresholds):
    """
    Profil de tentatives avec seuils dynamiques.
    """
    if pd.isna(nb_tentatives):
        return "inconnu"

    t1, t2 = thresholds["tentatives"]

    if nb_tentatives <= t1:
        return "resolution_rapide"
    if nb_tentatives <= t2:
        return "resolution_moyenne"
    return "resolution_longue"


def profil_temps_dynamic(mean_temps, nb_temps_valides, thresholds):
    """
    Profil de temps avec seuils dynamiques.
    Ignore le temps s'il n'est pas renseigné.
    """
    if nb_temps_valides == 0 or pd.isna(mean_temps):
        return "non_renseigne"

    t1, t2 = thresholds["temps"]

    if mean_temps <= t1:
        return "rapide"
    if mean_temps <= t2:
        return "modere"
    return "lent"


def profil_reussite_exercice(reussite_finale):
    if pd.isna(reussite_finale):
        return "inconnu"
    return "reussi" if reussite_finale == 1 else "non_reussi"


def profil_taux_reussite(taux_reussite):
    if pd.isna(taux_reussite):
        return "inconnu"

    if taux_reussite >= 0.75:
        return "forte_reussite"
    if taux_reussite >= 0.5:
        return "reussite_moyenne"
    return "faible_reussite"


def classe_finale_from_profils(profil_prog, profil_modif, profil_temps, profil_tent, profil_reussite=None):
    """
    Déduit une classe finale à partir des profils.

    Version moins restrictive : avant, presque toutes les règles étaient dans
    `if profil_temps == "non_renseigne"`, donc dès que le temps était présent,
    la fonction retournait très souvent `profil_intermediaire`.

    Ici, le temps sert comme information complémentaire, mais ne bloque plus
    toute la classification.
    """
    bonne_reussite = profil_reussite in [None, "inconnu", "reussi", "forte_reussite", "reussite_moyenne"]
    mauvaise_reussite = profil_reussite in ["non_reussi", "faible_reussite"]
    temps_rapide_ou_absent = profil_temps in ["rapide", "non_renseigne"]
    temps_lent = profil_temps == "lent"

    # 1) Cas en difficulté : priorité aux échecs / faibles réussites.
    if mauvaise_reussite:
        if profil_prog in ["stagnation", "regression"]:
            return "bloque"
        if profil_modif == "gros_restructurateur":
            return "explorateur_chaotique"
        if profil_tent == "resolution_longue" or temps_lent:
            return "en_difficulte"
        return "fragile"

    # 2) Bons profils : progression efficace avec peu de changements.
    if (
        bonne_reussite
        and profil_prog == "progression_rapide"
        and profil_modif == "petit_ajusteur"
        and profil_tent == "resolution_rapide"
        and temps_rapide_ou_absent
    ):
        return "expert_progressif"

    if (
        bonne_reussite
        and profil_prog in ["progression_rapide", "progression_lente"]
        and profil_modif in ["petit_ajusteur", "modificateur_modere"]
        and profil_tent in ["resolution_rapide", "resolution_moyenne"]
    ):
        return "reviseur_methodique"

    # 3) Progression avec grosses modifications : il restructure beaucoup, mais avance.
    if profil_prog in ["progression_rapide", "progression_lente"] and profil_modif == "gros_restructurateur":
        if profil_tent == "resolution_longue" or temps_lent:
            return "restructurateur_lent"
        return "gros_restructurateur_productif"

    # 4) Peu ou pas de progression.
    if profil_prog in ["stagnation", "regression"]:
        if profil_modif == "gros_restructurateur":
            return "explorateur_chaotique"
        if profil_tent == "resolution_longue" or temps_lent:
            return "bloque"
        return "stagne_mais_corrige"

    # 5) Cas intermédiaires plus informatifs que `profil_intermediaire`.
    if profil_prog == "progression_lente" and profil_tent == "resolution_longue":
        return "perseverant_lent"

    if profil_modif == "petit_ajusteur" and profil_tent == "resolution_longue":
        return "petits_pas_nombreux"

    if profil_modif == "modificateur_modere":
        return "apprenant_regulier"

    return "profil_intermediaire"


def distribution_classes(stats, col="classe"):
    """
    Retourne la distribution des classes en nombre et en pourcentage.
    Utile pour vérifier si la classification est trop uniforme.
    """
    counts = stats[col].value_counts(dropna=False)
    pct = stats[col].value_counts(normalize=True, dropna=False).mul(100).round(2)

    return (
        pd.DataFrame({"nb": counts, "%": pct})
        .reset_index()
        .rename(columns={"index": col})
    )


def diagnostic_classification(stats):
    """
    Diagnostic rapide pour comprendre pourquoi les classes sont concentrées.
    """
    colonnes = [
        "profil_progression",
        "profil_modification",
        "profil_tentatives",
        "profil_temps",
        "profil_reussite",
        "classe",
    ]

    return {
        col: distribution_classes(stats, col)
        for col in colonnes
        if col in stats.columns
    }


def pre_calcul(dfo, solution_df):
    """
    Construit un dataset allégé pour la classification.
    """
    dataset = build_transition_dataset(
        dfo,
        solution_df,
        include_code_errors=False,
        include_solution_scores=True,
        include_temps=True,
        include_status=True,
        include_codes=False,
        save_path="pre_calcul.json",
    )

    cols = [
        "id",
        "exercice",
        "type_exercice",
        "t",
        "t_plus_1",
        "distance_zss",
        "score_t_solution",
        "score_t_plus_1_solution",
        "progression_solution",
        "temps_t",
        "temps_t_plus_1",
        "delta_temps",
        "statut_t",
        "statut_t_plus_1",
        "reussite_finale_exercice",
    ]
    return dataset[[col for col in cols if col in dataset.columns]]


def aggregate_classification_stats(dataset, group_cols, success_mode):
    """
    Agrège les métriques nécessaires à la classification.

    success_mode:
    - "user" : calcule un taux de réussite par utilisateur.
    - "exercise" : garde la réussite finale du couple user/exercice.
    """
    df = dataset.copy()
    check_required_columns(
        df,
        [
            "id",
            "exercice",
            "progression_solution",
            "distance_zss",
            "t_plus_1",
            "delta_temps",
            "reussite_finale_exercice",
        ]
    )

    agg_dict = {
        "mean_progress": ("progression_solution", "mean"),
        "mean_zss": ("distance_zss", "mean"),
        "max_tentative": ("t_plus_1", "max"),
        "nb_transitions": ("id", "size"),
        "mean_delta_temps": ("delta_temps", "mean"),
        "nb_temps_valides": ("delta_temps", lambda s: s.notna().sum()),
    }

    if "type_exercice" in df.columns and "exercice" in group_cols:
        agg_dict["type_exercice"] = ("type_exercice", "first")

    stats = df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()

    if success_mode == "user":
        reussite_ex = (
            df.groupby(["id", "exercice"], dropna=False)
            .agg(reussite_finale_exercice=("reussite_finale_exercice", "max"))
            .reset_index()
        )

        success = (
            reussite_ex.groupby("id", dropna=False)
            .agg(
                taux_reussite=("reussite_finale_exercice", "mean"),
                nb_exercices=("exercice", "size"),
                nb_exercices_reussis=("reussite_finale_exercice", "sum"),
            )
            .reset_index()
        )
        stats = stats.merge(success, on="id", how="left")
        stats["profil_reussite"] = stats["taux_reussite"].apply(profil_taux_reussite)

    elif success_mode == "exercise":
        success = (
            df.groupby(group_cols, dropna=False)
            .agg(reussite_finale_exercice=("reussite_finale_exercice", "max"))
            .reset_index()
        )
        stats = stats.merge(success, on=group_cols, how="left")
        stats["profil_reussite"] = stats["reussite_finale_exercice"].apply(profil_reussite_exercice)

    else:
        raise ValueError("success_mode doit être 'user' ou 'exercise'")

    return stats


def apply_dynamic_profiles(stats):
    """
    Ajoute les profils dynamiques et la classe finale.
    """
    thresholds = compute_dynamic_thresholds(stats)

    stats = stats.copy()
    stats["profil_progression"] = stats["mean_progress"].apply(
        lambda x: profil_progression_dynamic(x, thresholds)
    )
    stats["profil_modification"] = stats["mean_zss"].apply(
        lambda x: profil_modification_dynamic(x, thresholds)
    )
    stats["profil_tentatives"] = stats["max_tentative"].apply(
        lambda x: profil_tentatives_dynamic(x, thresholds)
    )
    stats["profil_temps"] = stats.apply(
        lambda row: profil_temps_dynamic(
            row["mean_delta_temps"],
            row["nb_temps_valides"],
            thresholds,
        ),
        axis=1,
    )

    stats["classe"] = stats.apply(
        lambda row: classe_finale_from_profils(
            row["profil_progression"],
            row["profil_modification"],
            row["profil_temps"],
            row["profil_tentatives"],
            row["profil_reussite"],
        ),
        axis=1,
    )

    return stats, thresholds


def build_user_classification(dataset):
    """
    Classification globale par utilisateur.
    """
    stats = aggregate_classification_stats(
        dataset,
        group_cols=["id"],
        success_mode="user",
    )
    return apply_dynamic_profiles(stats)


def build_user_exercice_classification(dataset):
    """
    Classification par couple (utilisateur, exercice).
    """
    stats = aggregate_classification_stats(
        dataset,
        group_cols=["id", "exercice"],
        success_mode="exercise",
    )
    return apply_dynamic_profiles(stats)