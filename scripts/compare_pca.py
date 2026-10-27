"""Compare tuned SVM classifiers with and without PCA using the same split.

Run from the repository root:
    python scripts/compare_pca.py

The test set is used only once, after hyperparameter selection on training data.
"""
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "Crop_Recommendation.csv")
X = df.drop(columns="Crop")
y = df["Crop"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

experiments = {
    "SVM sans PCA": (
        Pipeline([("scaler", StandardScaler()), ("svm", SVC())]),
        {
            "svm__kernel": ["rbf", "linear"],
            "svm__C": [0.1, 1, 10],
            "svm__gamma": [0.01, 0.1, 1],
        },
    ),
    "SVM avec PCA": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("svm", SVC()),
        ]),
        {
            "pca__n_components": [2, 3, 5, 7],
            "svm__kernel": ["rbf", "linear"],
            "svm__C": [0.1, 1, 10],
            "svm__gamma": [0.01, 0.1, 1],
        },
    ),
}

for name, (pipeline, grid_params) in experiments.items():
    search = GridSearchCV(
        pipeline, grid_params, scoring="accuracy", cv=cv, n_jobs=-1
    )
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    print(f"\\n{name}")
    print("Best CV accuracy:", round(search.best_score_, 5))
    print("Best params:", search.best_params_)
    print("Test accuracy:", round(accuracy_score(y_test, pred), 5))
    print(classification_report(y_test, pred, zero_division=0))

print("\\nNote: these are new experiments; compare against the original notebook separately.")
