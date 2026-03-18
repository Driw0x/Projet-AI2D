import pandas as pd
import ast_error_detection as aed
import ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Outils AST

def code_to_ast(code):
    """
    Transforme un code Python sous forme de texte en AST.
    AST = arbre syntaxique abstrait.
    """
    return ast.parse(code)

def ast_dump(t):
    """
    Affiche l'AST de manière lisible.
    Utile pour voir la structure du code.
    """
    return ast.dump(t, indent=2)

# Lecture des donnees

def read_data(path):
    """
    Lit un fichier de donnees selon son extension.
    Formats geres :
    - json
    - csv
    """
    format = path.split(".")[-1]

    match format:
        case "json":
            df = pd.read_json(path)
        case "csv":
            df = pd.read_csv(path)
        case _:
            print("Format incorrect")
            return

    return df

# Preparation des donnees AlgoPython

def AlgoPython_data(df):
    """
    Transforme les donnees brutes AlgoPython en DataFrame plat.

    etapes :
    - eclate la colonne classes
    - eclate la colonne comptes
    - recupère les trajectoires
    - eclate les tentatives
    - transforme le statut en binaire :
        ok -> 1
        autres -> 0
    """
    # copie du DataFrame
    classes = df[df.columns]

    # une ligne par classe
    classes = classes.explode("classes", ignore_index=True)

    # normalise le contenu JSON de la colonne classes
    classes_details = pd.json_normalize(classes["classes"])
    classes = classes.drop(columns=["classes"])
    classes = classes.join(classes_details)

    # une ligne par compte
    comptes = classes.explode("comptes", ignore_index=True)

    # normalise le contenu JSON de la colonne comptes
    comptes_details = pd.json_normalize(comptes["comptes"], max_level=0)
    comptes = comptes.drop(columns=["comptes"])
    comptes = comptes.join(comptes_details)

    # recupère les trajectoires
    trajectories = comptes["trajectories"]

    # garde seulement les trajectoires qui sont des dictionnaires
    trajectories = trajectories.where(trajectories.map(type).eq(dict), {})

    # passe les trajectoires en format large
    traj_wide = trajectories.apply(pd.Series)

    # passe en format long
    traj_long = traj_wide.stack(future_stack=True).reset_index(name="tentatives")

    # recupère les colonnes de contexte
    contexte = comptes.drop(columns=["trajectories"])
    traj_long = traj_long.join(contexte, on="level_0")
    traj_long = traj_long.drop(columns=["level_0"])

    # une ligne par tentative
    traj_long = traj_long.explode("tentatives", ignore_index=True)

    # normalise les tentatives
    tentatives = pd.json_normalize(traj_long["tentatives"])

    # joint les infos des tentatives au contexte
    df_final = traj_long.drop(columns=["tentatives"]).join(tentatives)

    # enlève les lignes sans code
    df_final = df_final[df_final["code"].notna()]

    # remet un index propre
    df_final = df_final.reset_index(drop=True)

    # enlève les statuts "err" et "ask"
    df_final = df_final[(df_final["statut"] != "err") & (df_final["statut"] != "ask")]

    # transforme ok en 1 et le reste en 0
    df_final["statut"] = np.where(df_final["statut"] == "ok", 1, 0)

    return df_final

# Comparaison de programmes

def primary_code_error_two_prog(p1, p2):
    """
    Compare deux programmes et retourne les erreurs principales detectees.
    En cas d'erreur dans l'outil, retourne [0, []].
    """
    try:
        return aed.get_primary_code_errors(p1, p2)
    except:
        return [0, []]


def prog_vs_answer(p1, list_answer):
    """
    Compare un programme à une ou plusieurs reponses attendues.
    Retourne la typologie d'erreurs.
    En cas d'erreur, retourne [0, {}].
    """
    try:
        return aed.get_typology_based_code_error(p1, list_answer)
    except:
        return [0, {}]
    
def regle(primary_code_error):
    """
    Affiche une interpretation textuelle simple des erreurs detectees.
    """
    def extract_function_name(s):
        """
        Extrait le nom de la fonction à partir du contexte.
        """
        return s.split(":")[-1].strip()
    
    for i in primary_code_error[1]:
        erreur = i[0]
        context = i[-1].split(" > ")

        print(erreur)

        match erreur:
            case 'CONST_VALUE_MISMATCH':
                print(f"\tConstante en argument pour la fonction {extract_function_name(context[-2])} incorrecte.")

            case 'MISSING_CONST_VALUE':
                print(f"\tManque l'argument à la fonction {extract_function_name(context[-2])} dans le programme")

            case 'MISSING_CALL_STATEMENT' | 'MISSING_FOR_LOOP':
                print(f"\tManque l'appel à la fonction {extract_function_name(context[-1])} dans le programme")

            case _ if 'UNNECESSARY' in erreur:
                print(f"\tAppel inutile à la fonction {extract_function_name(context[-1])} dans le programme")

            case _:
                print("\tRetour pas encore prise en charge")

def comparaison_tentative_solution(df):
    """
    Compare toutes les tentatives d'un exercice à la dernière tentative,
    consideree ici comme la solution de reference.
    """
    prog = []
    p = df.reset_index(drop=True)

    # recupère tous les codes
    for i in range(len(p)):
        prog.append(p.iloc[i]["code"])

    # dernière tentative = reponse de reference
    answer = p.iloc[len(p) - 1]["code"]
    ast_answer = code_to_ast(answer)

    # compare chaque tentative à la reponse
    for i in range(len(prog) - 1):
        print(f"Tentative {i + 1}: ")

        if prog[i]:
            ast_p = code_to_ast(prog[i])
            regle(primary_code_error_two_prog(ast_p, ast_answer))
        else:
            print("Code vide")

def analyse_user(id_compte, df):
    """
    Analyse tous les exercices d'un utilisateur.
    Pour chaque exercice :
    - compare les tentatives si plusieurs essais
    - indique reussite directe ou abandon si un seul essai
    """
    u = df[(df["id_compte"] == id_compte)]
    ex = np.unique(u["level_1"])

    for e in ex:
        print(f"Exercice {e}:")
        p = u[u["level_1"] == e]

        if len(p) > 1:
            comparaison_tentative_solution(p)
        else:
            print("Qu'un seul essai")
            if p["statut"].iloc[0] == 1:
                print("Il a reussi du premier coup")
            else:
                print("Il a abandonne dès le premier essai")
        print()

# Pre calcul

def build_user_exercise_count_dict(df):
    """
    Calcule le nombre de tentatives pour chaque couple (user, exercice).

    Retour :
    - dict {(id_compte, exercice): nb_tentatives}
    """
    df = df[df["code"] != ""]

    counts = {}

    for user_id in np.unique(df["id_compte"]):
        user_df = df[df["id_compte"] == user_id]

        for ex in np.unique(user_df["level_1"]):
            key = (
                int(user_id) if isinstance(user_id, np.generic) else user_id,
                str(ex)
            )
            counts[key] = len(user_df[user_df["level_1"] == ex])

    return counts

def build_user_exercise_error_dict(df):
    """
    Calcule une seule fois les erreurs uniques pour chaque couple (user, exercice).

    Retour :
    - dict {(id_compte, exercice): [erreurs_uniques]}
    """
    df = df[df["code"] != ""]

    user_ex_errors = {}

    for user_id in np.unique(df["id_compte"]):
        user_df = df[df["id_compte"] == user_id]

        for ex in np.unique(user_df["level_1"]):
            sub = user_df[user_df["level_1"] == ex].reset_index(drop=True)

            if len(sub) > 1:
                errs = []
                for i in range(len(sub) - 1):
                    res = primary_code_error_two_prog(
                        sub["code"].iloc[i],
                        sub["code"].iloc[i + 1]
                    )[1]
                    for err in res:
                        errs.append(err[0])

                errs = list(np.unique(errs))
            else:
                errs = []

            key = (
                int(user_id) if isinstance(user_id, np.generic) else user_id,
                str(ex)
            )
            user_ex_errors[key] = errs

    return user_ex_errors
    
# Construction des dictionnaires

def recherche_echantillon(df, user_ex_errors, user_ex_counts):
    """
    Cherche les users ayant plus de 5 tentatives sur au moins un exercice.

    Retour :
    - u : liste des users retenus
    - c : nb de users par exercice
    - e : nb d'apparitions par erreur
    - ex_e : nb d'apparitions par couple (exercice, erreur)
    - ex_u : users par exercice
    """
    df = df[df["code"] != ""]

    if user_ex_errors is None:
        user_ex_errors = build_user_exercise_error_dict(df)

    if user_ex_counts is None:
        user_ex_counts = build_user_exercise_count_dict(df)

    u = []
    c = {}
    e = {}
    ex_e = {}
    ex_u = {}

    all_ex = [str(ex) for ex in np.unique(df["level_1"])]
    all_users = [
        int(user_id) if isinstance(user_id, np.generic) else user_id
        for user_id in np.unique(df["id_compte"])
    ]

    for ex in all_ex:
        ex_u[ex] = []

    for user_id in all_users:
        print(f"User {user_id}")
        u_add = False

        for ex in all_ex:
            key = (user_id, ex)
            nb_tentatives = user_ex_counts.get(key, 0)

            if nb_tentatives > 5:
                erreurs = user_ex_errors.get(key, [])

                ex_u[ex].append(user_id)
                u_add = True

                if ex not in c:
                    c[ex] = 0
                c[ex] += 1

                for err in erreurs:
                    if err not in e:
                        e[err] = 0
                    if (ex, err) not in ex_e:
                        ex_e[(ex, err)] = 0

                    ex_e[(ex, err)] += 1
                    e[err] += 1

                print(f"Exercice {ex}")
                print(f"Erreur: {erreurs}")

        if u_add:
            u.append(user_id)

        print()

    print(u)
    print(c)
    print(e)
    print(ex_e)
    print(ex_u)

    return u, c, e, ex_e, ex_u

def filtre_err(df, err, user_ex_errors, user_ex_counts):
    """
    Filtre les couples (user, exercice) ayant plus de 5 tentatives
    et contenant au moins une erreur de la liste err.
    - réutilise user_ex_errors
    - réutilise user_ex_counts

    Retour :
    - dict_err_ex : erreur -> exercices
    - dict_ex_err : exercice -> erreurs
    - dict_ex_id : exercice -> users
    - dict_id_ex : user -> exercices
    """
    df = df[df["code"] != ""]

    err = [str(e) for e in err]

    dict_ex_id = {}
    dict_id_ex = {}
    dict_err_ex = {}
    dict_ex_err = {}

    all_ex = [str(ex) for ex in np.unique(df["level_1"])]
    all_users = [
        int(user_id) if isinstance(user_id, np.generic) else user_id
        for user_id in np.unique(df["id_compte"])
    ]

    for user_id in all_users:
        for ex in all_ex:
            key = (user_id, ex)
            nb_tentatives = user_ex_counts.get(key, 0)

            if nb_tentatives > 5:
                eu = [str(e) for e in user_ex_errors.get(key, [])]

                has_selected_error = any(i in eu for i in err)

                if has_selected_error:
                    for e in eu:
                        if e not in dict_err_ex:
                            dict_err_ex[e] = []
                        if ex not in dict_err_ex[e]:
                            dict_err_ex[e].append(ex)

                        if ex not in dict_ex_err:
                            dict_ex_err[ex] = []
                        if e not in dict_ex_err[ex]:
                            dict_ex_err[ex].append(e)

                    if ex not in dict_ex_id:
                        dict_ex_id[ex] = []
                    if user_id not in dict_ex_id[ex]:
                        dict_ex_id[ex].append(user_id)

                    if user_id not in dict_id_ex:
                        dict_id_ex[user_id] = []
                    if ex not in dict_id_ex[user_id]:
                        dict_id_ex[user_id].append(ex)

    return dict_err_ex, dict_ex_err, dict_ex_id, dict_id_ex

# Visualisation

def clean_ex_err_dict(dict_ex_err):
    """
    Convertit les cles d'exercices en str et les erreurs en str.
    Supprime les doublons.
    """
    cleaned = {}
    for ex, errs in dict_ex_err.items():
        ex_clean = str(ex)
        errs_clean = sorted(set(str(err) for err in errs))
        cleaned[ex_clean] = errs_clean
    return cleaned

def clean_err_ex_dict(dict_err_ex):
    """
    Convertit les cles erreurs en str et les exercices en str.
    Supprime les doublons.
    """
    cleaned = {}
    for err, exs in dict_err_ex.items():
        err_clean = str(err)
        exs_clean = sorted(set(str(ex) for ex in exs))
        cleaned[err_clean] = exs_clean
    return cleaned

def count_exercises_per_error(dict_err_ex):
    """
    Retourne un dict : erreur -> nombre d'exercices concernes
    """
    dict_err_ex = clean_err_ex_dict(dict_err_ex)
    return {err: len(exs) for err, exs in dict_err_ex.items()}

def count_errors_per_exercise(dict_ex_err):
    """
    Retourne un dict : exercice -> nombre d'erreurs presentes
    """
    dict_ex_err = clean_ex_err_dict(dict_ex_err)
    return {ex: len(errs) for ex, errs in dict_ex_err.items()}

def plot_bar_dict(data_dict, title, xlabel, ylabel, figsize=(12, 5), sort_desc=True):
    """
    Affiche un barplot à partir d'un dictionnaire.
    """
    items = list(data_dict.items())
    if sort_desc:
        items = sorted(items, key=lambda x: x[1], reverse=True)

    labels = [str(k) for k, v in items]
    values = [v for k, v in items]

    plt.figure(figsize=figsize)
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def build_exercise_error_matrix(dict_ex_err):
    """
    Construit la matrice binaire exercice x erreur.
    Retourne :
    - M : matrice binaire numpy
    - ex_list : liste des exercices
    - err_list : liste des erreurs
    """
    dict_ex_err = clean_ex_err_dict(dict_ex_err)

    ex_list = sorted(dict_ex_err.keys())
    err_list = sorted(set(err for errs in dict_ex_err.values() for err in errs))

    ex_index = {ex: i for i, ex in enumerate(ex_list)}
    err_index = {err: j for j, err in enumerate(err_list)}

    M = np.zeros((len(ex_list), len(err_list)), dtype=int)

    for ex, errs in dict_ex_err.items():
        for err in errs:
            M[ex_index[ex], err_index[err]] = 1

    return M, ex_list, err_list

def jaccard_similarity_matrix(M):
    """
    Calcule la matrice de similarite de Jaccard entre lignes d'une matrice binaire.
    """
    n = M.shape[0]
    S = np.zeros((n, n), dtype=float)

    for i in range(n):
        A = M[i].astype(bool)
        for j in range(n):
            B = M[j].astype(bool)
            inter = np.logical_and(A, B).sum()
            union = np.logical_or(A, B).sum()
            S[i, j] = inter / union if union > 0 else 0.0

    return S

def plot_similarity_matrix(S, labels, title="Matrice de similarite entre exercices", figsize=(10, 8)):
    """
    Affiche une matrice de similarite.
    """
    plt.figure(figsize=figsize)
    plt.imshow(S, interpolation='nearest', aspect='auto')
    plt.colorbar(label='Similarite de Jaccard')
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def project_exercises_pca(M, n_components=2):
    """
    Projection PCA des exercices.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(M)

def cluster_exercises_2d(X_2d, n_clusters=4, random_state=42):
    """
    Clustering KMeans sur la projection 2D.
    Retourne :
    - labels
    - centers
    - kmeans
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_2d)
    centers = kmeans.cluster_centers_
    return labels, centers, kmeans

def make_cluster_grid(X_2d, model, grid_size=400, margin=1.0):
    """
    Cree la grille et predit les zones des clusters.
    """
    x_min, x_max = X_2d[:, 0].min() - margin, X_2d[:, 0].max() + margin
    y_min, y_max = X_2d[:, 1].min() - margin, X_2d[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    return xx, yy, Z

def plot_clusters_2d(X_2d, labels, ex_list, centers=None, xx=None, yy=None, Z=None,
                     title="Projection 2D des exercices avec frontières de clusters",
                     figsize=(12, 8)):
    """
    Affiche la projection 2D avec zones de clusters, points, labels et centres.
    """
    plt.figure(figsize=figsize)

    if xx is not None and yy is not None and Z is not None:
        plt.contourf(xx, yy, Z, alpha=0.2)

    n_clusters = len(np.unique(labels))

    for cluster_id in range(n_clusters):
        pts = X_2d[labels == cluster_id]
        plt.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {cluster_id}", s=60)

    for i, ex in enumerate(ex_list):
        plt.text(X_2d[i, 0], X_2d[i, 1], ex, fontsize=9)

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=120, linewidths=2, label='Centres')

    plt.title(title)
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_error_visualizations(dict_err_ex, dict_ex_err, n_clusters=4):
    """
    Lance toute la visualisation demandee :
    - nb exercices par erreur
    - nb erreurs par exercice
    - matrice de similarite
    - projection 2D avec clusters
    """
    err_counts = count_exercises_per_error(dict_err_ex)
    ex_counts = count_errors_per_exercise(dict_ex_err)

    plot_bar_dict(
        err_counts,
        title="Nombre d'exercices par erreur",
        xlabel="Erreur",
        ylabel="Nb exercices",
        figsize=(14, 6)
    )

    plot_bar_dict(
        ex_counts,
        title="Nombre d'erreurs par exercice",
        xlabel="Exercice",
        ylabel="Nb erreurs",
        figsize=(10, 5)
    )

    M, ex_list, err_list = build_exercise_error_matrix(dict_ex_err)

    S = jaccard_similarity_matrix(M)
    plot_similarity_matrix(S, ex_list)

    X_2d = project_exercises_pca(M)
    labels, centers, model = cluster_exercises_2d(X_2d, n_clusters=n_clusters)
    xx, yy, Z = make_cluster_grid(X_2d, model)

    plot_clusters_2d(X_2d, labels, ex_list, centers=centers, xx=xx, yy=yy, Z=Z)