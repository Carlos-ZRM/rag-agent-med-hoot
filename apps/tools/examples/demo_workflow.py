"""
End-to-end demo: EDA -> preprocessing -> train -> tune -> persist.

Run from apps/tools/:
    python examples/demo_workflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script without installing the package.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ds_framework.eda import (
    correlacion,
    desc_table,
    detectar_outliers,
    freq,
    tipo_dato,
)
from ds_framework.models import (
    ModelTrainer,
    grid_search,
    load_model,
    save_model,
)
from ds_framework.preprocessing import (
    build_preprocessor,
    rename_column,
    split_train_test,
)


def load_data() -> pd.DataFrame:
    data = load_iris(as_frame=True)
    df = data.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)":  "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)":  "petal_width",
        }
    )
    # add a synthetic categorical column for demo purposes
    df["origin"] = pd.Categorical(
        ["A", "B", "C"] * (len(df) // 3 + 1)
    )[: len(df)]
    return df


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    df = load_data()
    target = "target"

    # ---------- EDA ----------
    section("1. Descriptive statistics")
    print(
        desc_table(
            df,
            ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            tipo="num",
        )
    )

    section("2. Frequency of categorical column 'origin'")
    for table in freq(df, "origin"):
        print(table)

    section("3. Data-type inspection")
    print(tipo_dato(df))

    section("4. Correlation matrix (Pearson)")
    print(
        correlacion(
            df,
            ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            flag_threshold=0.7,
        )
    )

    section("5. Outlier detection")
    _, resumen, _ = detectar_outliers(
        df,
        variables=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        usar_isoforest=True,
        contamination=0.05,
    )
    print(resumen)

    # ---------- Preprocessing ----------
    section("6. Column renaming")
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        rename_column(df, col, tipo="numerica")
    rename_column(df, "origin", tipo="categorica")

    numeric_cols = [c for c in df.columns if c.startswith("c_")]
    categorical_cols = [c for c in df.columns if c.startswith("v_")]
    print("numeric_cols     =", numeric_cols)
    print("categorical_cols =", categorical_cols)

    section("7. Train/test split")
    X_train, X_test, y_train, y_test = split_train_test(
        df, target=target, stratify=True, random_state=42
    )
    print(f"X_train={X_train.shape}  X_test={X_test.shape}")

    # ---------- Model training ----------
    section("8. Train and evaluate")
    pre = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    pipe = Pipeline(
        [("pre", pre), ("clf", RandomForestClassifier(random_state=42))]
    )

    trainer = ModelTrainer(pipe)
    trainer.fit(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    print("accuracy :", round(metrics["accuracy"], 4))
    print("f1       :", round(metrics["f1"], 4))
    print("roc_auc  :", round(metrics.get("roc_auc", float("nan")), 4))

    # ---------- Hyperparameter tuning ----------
    section("9. Grid search (small grid)")
    gs = grid_search(
        pipe,
        param_grid={
            "clf__n_estimators": [50, 150],
            "clf__max_depth": [None, 5],
        },
        X=X_train,
        y=y_train,
        cv=3,
        scoring="f1_weighted",
    )
    print("best_params :", gs.best_params_)
    print("best_score  :", round(gs.best_score_, 4))

    # ---------- Persistence ----------
    section("10. Save/load model")
    artifacts = HERE.parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    path = save_model(
        gs.best_estimator_,
        artifacts / "iris_rf.joblib",
        metadata={
            "best_params": gs.best_params_,
            "metrics": {
                k: v for k, v in metrics.items() if k not in {"report", "confusion_matrix"}
            },
        },
    )
    print(f"Saved to {path}")

    loaded, meta = load_model(path, with_metadata=True)
    print("Loaded model:", type(loaded).__name__)
    print("Metadata keys:", list(meta))


if __name__ == "__main__":
    main()
