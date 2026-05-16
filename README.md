# Projet AI2D — Analyse de trajectoires AlgoPython

Projet Python d'analyse des trajectoires de code d'étudiants sur des exercices AlgoPython.

L'objectif est de transformer les données brutes en datasets exploitables pour étudier :

- l'évolution d'un code entre deux tentatives successives ;
- les distances structurelles entre programmes avec les AST Python ;
- les erreurs détectées par `ast_error_detection` ;
- la progression vers une solution attendue ;
- les profils d'apprentissage des étudiants.

---

## Structure du projet

```text
Projet-AI2D/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── 2025.json              # Données brutes AlgoPython, non fournies dans le dépôt
│   ├── exercises.json         # Solutions attendues, non fournies dans le dépôt
│   ├── cas1.json              # Généré par le projet
│   ├── cas2.json              # Généré par le projet
│   └── pre_calcul.json        # Généré par le projet
│
├── models/
│   ├── differences_tag.py
│   └── evolution_code.py
│
├── tools/
│   ├── __init__.py
│   ├── ast_tools.py
│   ├── cas.py
│   ├── classification.py
│   ├── comparaison.py
│   ├── dataset_builder.py
│   ├── io_tools.py
│   └── visualisations.py
│
└── tests/
    ├── __init__.py
    ├── all.py
    ├── conftest.py
    ├── test_ast_tools.py
    ├── test_cas.py
    ├── test_classification.py
    ├── test_comparaison.py
    ├── test_dataset_builder.py
    └── test_io_tools.py
```

> Le dossier `data/` peut être absent du dépôt si les données sont privées. Dans ce cas, il faut le créer localement avant de lancer le pipeline principal.

---

## Rôle des dossiers

### `models/`

Contient les classes simples utilisées pour représenter les résultats d'analyse :

- `Differences_tag` : booléens indiquant les types de modifications détectées entre deux codes ;
- `Evolution_code` : informations sur l'apparition et les plages de modification des structures de contrôle.

### `tools/`

Contient les fonctions principales du projet, découpées par responsabilité.

| Fichier | Rôle |
|---|---|
| `io_tools.py` | Lecture, nettoyage et mise à plat des données AlgoPython |
| `ast_tools.py` | Parsing Python, AST, wrapper compatible distance ZSS |
| `comparaison.py` | Comparaison entre deux codes, erreurs AED, distance AST |
| `dataset_builder.py` | Construction des datasets de transitions `t -> t+1` |
| `cas.py` | Génération des datasets `cas1` et `cas2` |
| `classification.py` | Calcul des profils utilisateurs et utilisateur/exercice |
| `visualisations.py` | Graphiques d'analyse statistique |

### `tests/`

Contient les tests unitaires du projet.

Chaque fichier de test correspond à un module de `tools/` :

| Fichier de test | Module testé |
|---|---|
| `test_ast_tools.py` | `tools.ast_tools` |
| `test_cas.py` | `tools.cas` |
| `test_classification.py` | `tools.classification` |
| `test_comparaison.py` | `tools.comparaison` |
| `test_dataset_builder.py` | `tools.dataset_builder` |
| `test_io_tools.py` | `tools.io_tools` |

Le fichier `tests/conftest.py` ajoute un faux module `ast_error_detection` si la vraie librairie n'est pas installée. Cela permet de lancer les tests unitaires même sur une machine où la dépendance externe AED est absente.

---

## Pipeline général

```text
Données brutes AlgoPython
        ↓
Nettoyage et normalisation
        ↓
Extraction des trajectoires étudiant/exercice
        ↓
Construction des transitions t -> t+1
        ↓
Analyse AST et distance ZSS
        ↓
Détection d'erreurs de code
        ↓
Scores de progression vers la solution
        ↓
Classification des profils
        ↓
Visualisations statistiques
```

---

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Driw0x/Projet-AI2D.git
cd Projet-AI2D
```

### 2. Créer un environnement virtuel

Windows :

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Données attendues

Le pipeline principal attend les fichiers suivants :

```text
data/2025.json
data/exercises.json
```

Les fichiers générés par les analyses sont sauvegardés dans `data/` :

```text
data/cas1.json
data/cas2.json
data/pre_calcul.json
```

---

## Lancer le projet

```bash
python main.py
```

Le fichier `main.py` sert de point d'entrée. Il orchestre :

1. le chargement des données ;
2. la préparation du DataFrame AlgoPython ;
3. la génération ou le rechargement des datasets intermédiaires ;
4. la classification ;
5. l'affichage de tableaux et graphiques.

Certaines étapes sont commentées dans `main.py` pour pouvoir activer uniquement l'analyse voulue.

---

## Lancer les tests

Avec pytest directement :

```bash
pytest -v
```

Ou avec le lanceur fourni :

```bash
python tests/all.py
```

Les tests vérifient notamment :

- le parsing AST ;
- la mise à plat des données JSON ;
- la construction des transitions ;
- les fonctions de comparaison ;
- la génération de `cas1` et `cas2` ;
- la classification des profils.

---

## Remarques importantes

- `ast_error_detection` est une dépendance externe centrale pour les comparaisons détaillées.
- Les tests unitaires peuvent fonctionner sans la vraie librairie grâce au mock présent dans `tests/conftest.py`.
- Pour obtenir les vrais résultats d'analyse, il faut installer correctement `ast_error_detection` et disposer des données AlgoPython réelles.
- Les dossiers `__pycache__/` et `.pytest_cache/` ne doivent pas être versionnés.

---

## Exemples de questions analysables

Le projet permet d'étudier des questions comme :

- Un étudiant progresse-t-il entre deux tentatives ?
- Les corrections sont-elles de petits ajustements ou de grosses restructurations ?
- Quels exercices provoquent les plus grandes distances ZSS ?
- Quels types d'erreurs sont les plus fréquents ?
- Quels profils d'apprentissage apparaissent dans les données ?
