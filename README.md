# Classification de cultures agricoles avec SVM et PCA

**Projet de Machine Learning supervisé | Python · scikit-learn · Pandas · Matplotlib**

## Objectif
Prédire une **classe de culture agricole** à partir de sept caractéristiques du sol et du climat : azote (N), phosphore (P), potassium (K), température, humidité, pH et précipitations. Il s'agit d'un exercice de **classification multiclasse** sur un jeu de données existant, et non d'une recommandation agronomique validée sur le terrain.

## Données
- Fichier : `data/Crop_Recommendation.csv`
- **2 200 observations**, **22 classes**, **100 observations par classe**.
- Variables explicatives : `Nitrogen`, `Phosphorus`, `Potassium`, `Temperature`, `Humidity`, `pH_Value`, `Rainfall`.
- Cible : `Crop`.
- Aucune valeur manquante relevée dans le notebook.

L'équilibre des classes facilite l'évaluation, mais ne garantit ni l'absence de biais ni la représentativité du jeu de données pour des exploitations réelles. La provenance et les conditions de collecte doivent être précisées avant toute utilisation opérationnelle.

## Méthodologie
1. Inspection des données et visualisation exploratoire avec **t-SNE**.
2. Séparation **entraînement/test : 80 % / 20 %**, stratifiée sur la culture (`random_state=42`).
3. **Modèle de référence** : `StandardScaler → SVC()` (paramètres par défaut, SVM à marge souple).
4. **Recherche d'hyperparamètres** : pipeline `StandardScaler → PCA → SVC`, avec `GridSearchCV` à **5 plis**, sur le seul ensemble d'entraînement.
5. Évaluation sur le jeu de test : accuracy, precision, recall, F1-score et matrices de confusion.
6. Sauvegarde du meilleur pipeline entraîné dans `svm_crop_model.pkl`.

**Remarque :** la projection t-SNE sert uniquement à l'exploration et n'est pas utilisée comme entrée du modèle. Elle ne prouve pas à elle seule la séparabilité ou la généralisation des classes.

## Résultats enregistrés dans le notebook

| Configuration | Accuracy sur le jeu de test |
| --- | ---: |
| SVM de référence (`StandardScaler → SVC()`) | **98,41 %** |
| SVM optimisé (`StandardScaler → PCA → SVC`) | **99,09 %** |

Meilleurs paramètres issus de `GridSearchCV` :
- `pca__n_components = 7`
- `svm__kernel = "rbf"`
- `svm__C = 1`
- `svm__gamma = 1`
- **Accuracy moyenne en validation croisée : 98,69 %**

Ces chiffres proviennent des sorties enregistrées dans `svm.ipynb` ; ils n'ont pas encore été revérifiés dans un nouvel environnement.

**Point méthodologique à approfondir :** avec **7 variables d'entrée et 7 composantes PCA**, cette configuration transforme les variables mais ne réduit pas leur nombre. Une comparaison contrôlée entre SVM avec et sans PCA est prévue afin d'en mesurer l'utilité.

## Structure du dépôt
```text
.
├── data/
│   └── Crop_Recommendation.csv
├── svm.ipynb                 # Analyse et entraînement d'origine
├── svm_crop_model.pkl       # Pipeline enregistré
├── A propos du dataset.pdf  # Documentation fournie avec le projet
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## Installation et exécution
Python 3.10+ recommandé.

```bash
git clone https://github.com/fatoundour/Machine-Learning-project.git
cd Machine-Learning-project
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook svm.ipynb
```

Lancez Jupyter depuis la racine du dépôt : le notebook charge `data/Crop_Recommendation.csv` avec un chemin relatif.

## Limites et améliorations prévues
- Comparer les modèles optimisés **avec et sans PCA**, sur la même séparation et avec une validation croisée identique.
- Vérifier la reproductibilité des résultats et les versions des dépendances.
- Documenter précisément l'origine, les unités et les conditions de collecte des données.
- Ajouter un script de prédiction et des tests.
- Éventuellement développer une démonstration interactive.

> Les performances présentées concernent ce jeu de données et ne constituent pas une validation agronomique en conditions réelles.
