# Projet-AI2D

Projet de recherche et d’analyse des trajectoires de code d’étudiants sur des plateformes d’apprentissage de programmation comme AlgoPython.

Le projet a pour objectif d’analyser l’évolution des programmes écrits par les étudiants entre plusieurs tentatives de résolution d’exercices afin d’identifier :

- les stratégies de correction ;
- les types d’erreurs fréquentes ;
- les profils d’apprentissage ;
- les comportements de progression ;
- les modifications structurelles dans le code.

Le pipeline repose principalement sur l’analyse d’AST Python, la distance de Zhang-Shasha, ainsi que des outils de détection d’erreurs syntaxiques et structurelles.

---

# Objectifs du projet

Le projet vise à :

- transformer des datasets éducatifs complexes en données exploitables ;
- reconstruire les trajectoires de résolution d’exercices ;
- comparer deux versions successives d’un programme ;
- détecter automatiquement les erreurs de code ;
- mesurer les évolutions structurelles des programmes ;
- classifier les profils d’apprentissage des étudiants.

---

# Fonctionnalités principales

## Préparation des données AlgoPython

Le pipeline transforme les données brutes en DataFrame pandas exploitables :

- explosion des structures JSON imbriquées ;
- extraction des classes, comptes et trajectoires ;
- normalisation des tentatives ;
- nettoyage des données ;
- binarisation des statuts de réussite (`ok -> 1`, sinon `0`).

Fonction principale :

```python
AlgoPython_data(df)
```

---

## Analyse AST Python

Le projet utilise le module `ast` de Python pour :

- parser les programmes ;
- construire des arbres syntaxiques abstraits ;
- comparer les structures de code ;
- préparer les distances AST.

Fonctions principales :

```python
code_to_ast(code)
ast_dump(tree)
```

---

## Distance structurelle entre programmes

Le projet mesure les différences entre deux programmes via :

- la distance de Zhang-Shasha ;
- les opérations de transformation AST ;
- la comparaison t → t+1.

Fonctions principales :

```python
compare_transition(...)
distance(...)
```

---

## Détection d’erreurs de code

Le projet s’appuie sur la librairie :

- `ast_error_detection`

pour identifier :

- erreurs d’appels de fonctions ;
- erreurs de boucles ;
- erreurs de conditions ;
- erreurs d’assignation ;
- erreurs d’opérations.

Fonctions principales :

```python
primary_code_error_two_prog(p1, p2)
prog_vs_answer(p1, list_answer)
```

---

## Génération de datasets de transitions

Le pipeline construit automatiquement les transitions :

```text
tentative t  -> tentative t+1
```

avec :

- distance AST ;
- progression ;
- temps ;
- statut ;
- erreurs détectées ;
- comparaison aux solutions attendues.

Fonction principale :

```python
build_transition_dataset(...)
```

---

## Cas d’étude

### Cas 1 — Comparaison brute des programmes

Dataset contenant :

- distance ZSS ;
- opérations AST ;
- erreurs principales ;
- typologies d’erreurs ;
- codes t et t+1.

Fonction :

```python
cas1(...)
```

---

### Cas 2 — Génération de commentaires pédagogiques

Transformation automatique des erreurs détectées en phrases interprétables.

Exemples :

- « Ajout d’une boucle for »
- « Suppression d’un appel à la fonction »
- « Changement de constante »

Fonction :

```python
cas2(...)
```

---

## Classification des profils étudiants

Le projet génère des profils dynamiques basés sur :

- progression ;
- temps ;
- nombre de tentatives ;
- restructuration du code ;
- réussite des exercices.

Exemples de profils :

- `expert_progressif`
- `reviseur_methodique`
- `explorateur_chaotique`
- `bloque`
- `perseverant_lent`

Fonctions principales :

```python
build_user_classification(...)
build_user_exercice_classification(...)
```

---

# Structure du projet

```text
Projet-AI2D/
│
├── data/
│   ├── 2025.json
│   ├── exercises.json
│   ├── cas1.json
│   ├── cas2.json
│   └── pre_calcul.json
│
├── utils.py
├── test.py
├── requirements.txt
└── .gitignore
```

---

# Pipeline général

```text
Données AlgoPython
        ↓
Nettoyage / Normalisation
        ↓
Extraction des trajectoires
        ↓
Comparaison t -> t+1
        ↓
Analyse AST
        ↓
Détection d’erreurs
        ↓
Mesure de progression
        ↓
Classification des profils
```

---

# Technologies utilisées

## Langage

- Python 3.11.9

## Librairies principales

- pandas
- numpy
- tqdm
- matplotlib
- ast
- pandasgui
- ast_error_detection

---

# Installation

## Cloner le projet

```bash
git clone https://github.com/Driw0x/Projet-AI2D.git
cd Projet-AI2D
```

---

## Créer un environnement virtuel

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / MacOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# Lancer le projet

```bash
python test.py
```

Le script principal permet :

- le chargement des datasets ;
- la génération des cas d’étude ;
- le pré-calcul ;
- la classification ;
- l’analyse détaillée utilisateur/exercice.

---

# Exemples d’analyses possibles

Le projet permet notamment de répondre à des questions comme :

- Un étudiant progresse-t-il réellement entre deux tentatives ?
- Corrige-t-il par petits ajustements ou grosses restructurations ?
- Quels types d’erreurs sont les plus fréquents ?
- Quels profils d’apprentissage émergent ?
- Combien de tentatives sont nécessaires selon les exercices ?
- Les étudiants qui restructurent beaucoup réussissent-ils davantage ?
