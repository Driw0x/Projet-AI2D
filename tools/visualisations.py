import matplotlib.pyplot as plt


def plot_distribution_distance_zss(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["distance_zss"].dropna(), bins=30)
    plt.title("Distribution des distances ZSS")
    plt.xlabel("Distance ZSS")
    plt.ylabel("Nombre de transitions")
    plt.tight_layout()
    plt.show()


def plot_progression_by_attempt(df):
    progress = (
        df.groupby("t_plus_1")["progression_solution"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.plot(progress["t_plus_1"], progress["progression_solution"], marker="o")
    plt.title("Progression moyenne selon le numéro de tentative")
    plt.xlabel("Tentative")
    plt.ylabel("Progression moyenne")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_zss_by_attempt(df):
    zss = (
        df.groupby("t_plus_1")["distance_zss"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.plot(zss["t_plus_1"], zss["distance_zss"], marker="o")
    plt.title("Distance ZSS moyenne selon le numéro de tentative")
    plt.xlabel("Tentative")
    plt.ylabel("Distance ZSS moyenne")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_attempts_by_exercise(df):
    attempts = (
        df.groupby(["id", "exercice"])["t_plus_1"]
        .max()
        .reset_index()
        .groupby("exercice")["t_plus_1"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 5))
    attempts.plot(kind="bar")
    plt.title("Nombre moyen de tentatives par exercice")
    plt.xlabel("Exercice")
    plt.ylabel("Nombre moyen de tentatives")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_success_rate_by_exercise(df):
    success = (
        df.groupby(["id", "exercice"])["reussite_finale_exercice"]
        .max()
        .reset_index()
        .groupby("exercice")["reussite_finale_exercice"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 5))
    success.plot(kind="bar")
    plt.title("Taux de réussite final par exercice")
    plt.xlabel("Exercice")
    plt.ylabel("Taux de réussite")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_zss_by_exercise(df):
    zss = (
        df.groupby("exercice")["distance_zss"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 5))
    zss.plot(kind="bar")
    plt.title("Distance ZSS moyenne par exercice")
    plt.xlabel("Exercice")
    plt.ylabel("Distance ZSS moyenne")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_zss_vs_solution(df):
    """
    Compare la distance ZSS avec la progression vers la solution.
    """

    data = df[
        ["distance_zss", "progression_solution"]
    ].dropna()

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data["distance_zss"],
        data["progression_solution"],
        alpha=0.3
    )

    plt.title("Distance ZSS vs progression vers la solution")
    plt.xlabel("Distance ZSS")
    plt.ylabel("Progression solution")

    plt.tight_layout()
    plt.show()


def run_all_plots(df):
    plot_distribution_distance_zss(df)
    plot_progression_by_attempt(df)
    plot_zss_by_attempt(df)
    plot_attempts_by_exercise(df)
    plot_zss_by_exercise(df)
    plot_success_rate_by_exercise(df)
    plot_zss_vs_solution(df)
