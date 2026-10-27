# Classification de cultures agricoles : SVM, PCA et sélection d'un modèle parcimonieux

**Machine Learning supervisé · Classification multiclasse · Python · scikit-learn**

## Présentation du projet

Ce projet étudie la classification de **22 types de cultures agricoles** à partir de caractéristiques du sol et du climat. L'objectif est de construire un classifieur SVM performant, puis de **questionner l'utilité de la réduction de dimension par PCA** à travers une comparaison expérimentale avec et sans PCA.

La démarche ne se limite donc pas à maximiser l'accuracy : elle cherche également à **justifier les transformations utilisées et à privilégier la solution la plus parcimonieuse lorsque les performances sont équivalentes**.

## Jeu de données

Le fichier `data/Crop_Recommendation.csv` contient **2 200 observations**, réparties équitablement entre **22 cultures** (100 observations par classe).

Les sept variables explicatives sont :

- **Sol :** azote (`Nitrogen`), phosphore (`Phosphorus`), potassium (`Potassium`) et pH (`pH_Value`).
- **Climat :** température (`Temperature`), humidité (`Humidity`) et précipitations (`Rainfall`).
- **Cible :** type de culture (`Crop`).

Le notebook ne relève aucune valeur manquante. L'équilibre des classes facilite l'analyse des performances par culture, sans garantir à lui seul la représentativité de données agricoles réelles.

## Démarche méthodologique

### 1. Exploration et préparation des données

- Inspection des variables et de la répartition des 22 classes.
- Visualisation exploratoire des observations avec **t-SNE**.
- Séparation stratifiée en **80 % entraînement / 20 % test** (`random_state=42`).
- Standardisation des variables numériques avec `StandardScaler`, intégré aux pipelines pour éviter d'ajuster le prétraitement sur le jeu de test.

### 2. Première modélisation : SVM et PCA

Le notebook `svm.ipynb` présente :

- Un **SVM de référence** avec les paramètres par défaut.
- Un pipeline **StandardScaler → PCA → SVM**.
- Une recherche des hyperparamètres par **GridSearchCV**, avec validation croisée à cinq plis.
- Une évaluation par accuracy, precision, recall, F1-score et matrices de confusion.

**Résultats du notebook initial :**

| Modèle | Accuracy sur le jeu de test |
| --- | ---: |
| SVM de référence (`StandardScaler → SVC()`) | **98,41 %** |
| SVM optimisé avec PCA (`StandardScaler → PCA → SVC`) | **99,09 %** |

La recherche initiale a sélectionné **7 composantes PCA**, un noyau **RBF**, `C=1` et `gamma=1` ; le score moyen en validation croisée est de **98,69 %**.

**Interprétation :** avec sept variables d'entrée et sept composantes conservées, la PCA change la représentation des données mais **ne réduit pas leur dimension** dans cette configuration optimale. Cela motive une expérience supplémentaire pour vérifier si cette transformation est réellement nécessaire.

### 3. Étude comparative : avec PCA ou sans PCA ?

Le script `scripts/compare_pca.py` reprend le problème de classification avec **deux pipelines distincts**, évalués selon la même séparation entraînement/test et une validation croisée stratifiée à cinq plis :

1. **Sans PCA :** `StandardScaler → SVM`.
2. **Avec PCA :** `StandardScaler → PCA → SVM`.

Dans les deux cas, une recherche d'hyperparamètres est effectuée sur les données d'entraînement avant l'évaluation finale sur le jeu de test. Les composantes PCA candidates sont **2, 3, 5 et 7**.

Les sorties de cette expérience sont conservées dans **`scripts/compare_pca_result.txt`**.

### 4. Résultats de la comparaison

| Critère | SVM sans PCA | SVM avec PCA |
| --- | ---: | ---: |
| Accuracy moyenne en validation croisée | **98,58 %** | **98,58 %** |
| Accuracy sur le jeu de test | **98,86 %** | **98,86 %** |
| Noyau sélectionné | Linéaire | Linéaire |
| `C` sélectionné | 10 | 10 |
| Composantes PCA sélectionnées | — | 7 |
| Prédictions correctes sur le test | 435 / 440 | 435 / 440 |

Les deux approches obtiennent les **mêmes performances mesurées** dans cette expérience. La configuration sélectionnée avec PCA conserve toutes les dimensions initiales.

Ces résultats correspondent à **une nouvelle expérience de comparaison** ; ils ne remplacent pas le score de **99,09 %** obtenu précédemment dans le notebook, dont l'espace de recherche et la procédure de validation diffèrent.

### 5. Interprétation et choix du modèle

L'étude montre que **l'ajout de la PCA ne procure aucun gain d'accuracy observé dans la comparaison contrôlée**. Puisque les deux configurations atteignent les mêmes scores et que la PCA retenue conserve les sept composantes, le pipeline **StandardScaler → SVM sans PCA** est le choix **le plus parcimonieux pour cette expérience** :

- Moins d'étapes de transformation.
- Même performance mesurée.
- Pipeline plus simple à comprendre et à maintenir.

La PCA a donc pleinement joué son rôle dans la démarche : **elle a été expérimentée, évaluée et confrontée à une alternative plus simple**. Le choix final repose sur les résultats, et non sur une préférence a priori.

Cette conclusion porte sur les expériences documentées ; elle ne démontre pas que la PCA serait inutile pour tout autre jeu de données ou toute autre configuration.

## Organisation du dépôt

```text
Machine-Learning-project/
├── data/
│   └── Crop_Recommendation.csv
├── scripts/
│   ├── compare_pca.py           # Expérience comparative avec/sans PCA
│   └── compare_pca_result.txt   # Sorties de l'expérience
├── svm.ipynb                     # Exploration, modélisation et évaluation initiales
├── svm_crop_model.pkl            # Modèle sérialisé présent dans le dépôt
├── A propos du dataset.pdf
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## Reproduire les expériences

Depuis la racine du dépôt :

```bash
git clone https://github.com/fatoundour/Machine-Learning-project.git
cd Machine-Learning-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Notebook initial :**

```bash
jupyter notebook svm.ipynb
```

**Comparaison avec/sans PCA :**

```bash
python3 scripts/compare_pca.py
```

Les résultats enregistrés sont consultables dans `scripts/compare_pca_result.txt`.

## Conclusion

Ce projet illustre une démarche de Machine Learning qui associe **exploration des données, modélisation SVM, optimisation des hyperparamètres et évaluation critique de la réduction de dimension**.

L'expérimentation complémentaire permet de **justifier le choix d'un modèle plus parcimonieux**, sans PCA, lorsque les performances mesurées sont identiques. Les résultats restent propres au jeu de données et au protocole d'évaluation utilisés ; ils ne constituent pas une validation agronomique en conditions réelles.
