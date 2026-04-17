# ds_framework

A small, opinionated data-science toolkit covering exploratory data
analysis, preprocessing, and machine-learning workflows built on top of
pandas and scikit-learn.

## Layout

```
apps/tools/
├── ds_framework/
│   ├── eda/                EDA: descriptive stats, correlation, outliers
│   ├── preprocessing/      Column-naming conventions & feature pipelines
│   ├── models/             Train / predict / evaluate / persist / tune
│   └── utils/              Display helpers
├── examples/
│   └── demo_workflow.py    End-to-end demo on the Iris dataset
├── requirements.txt
└── README.md
```

## Install

From the `apps/tools` directory:

```bash
pip install -r requirements.txt
```

Then import the package (run scripts from `apps/tools/` or add it to
`PYTHONPATH`):

```python
import ds_framework as dsf
```

## Quick start

### Exploratory Data Analysis

```python
import pandas as pd
from ds_framework.eda import desc_table, freq, tipo_dato, correlacion, detectar_outliers

df = pd.read_csv("data.csv")

desc_table(df, ["age", "income"], tipo="num")   # percentiles + %nulls
freq(df, "segment")                              # FA / FR / cumulative
tipo_dato(df)                                    # per-column type distribution
correlacion(df, ["age", "income", "tenure"], flag_threshold=0.7)

df_out, resumen, scores = detectar_outliers(
    df, variables=["age", "income"], usar_isoforest=True
)
```

### Preprocessing

```python
from ds_framework.preprocessing import rename_column, split_train_test, build_preprocessor

rename_column(df, "age",    tipo="numerica")     # -> c_age
rename_column(df, "gender", tipo="categorica")   # -> v_gender

X_train, X_test, y_train, y_test = split_train_test(df, target="label", stratify=True)

pre = build_preprocessor(
    numeric_cols=["c_age", "c_income"],
    categorical_cols=["v_gender"],
)
```

### Train, evaluate, persist

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from ds_framework.models import ModelTrainer, save_model, load_model

pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=42))])

trainer = ModelTrainer(pipe)
trainer.fit(X_train, y_train)
metrics = trainer.evaluate(X_test, y_test)
print(metrics)

save_model(trainer.model, "artifacts/rf.joblib", metadata={"metrics": metrics})
model = load_model("artifacts/rf.joblib")
```

### Hyperparameter tuning

```python
from ds_framework.models import grid_search

gs = grid_search(
    pipe,
    param_grid={"clf__n_estimators": [100, 300], "clf__max_depth": [None, 8]},
    X=X_train,
    y=y_train,
    cv=5,
    scoring="f1_weighted",
)
print(gs.best_params_, gs.best_score_)
```

## Running the demo

```bash
cd apps/tools
python examples/demo_workflow.py
```

## Public API

| Module | Functions |
|---|---|
| `ds_framework.eda` | `desc_table`, `freq`, `tipo_dato`, `correlacion`, `detectar_outliers` |
| `ds_framework.preprocessing` | `rename_column`, `NAMES`, `split_train_test`, `scale_features`, `encode_categoricals`, `build_preprocessor` |
| `ds_framework.models` | `ModelTrainer`, `train_model`, `predict`, `evaluate`, `save_model`, `load_model`, `grid_search`, `random_search` |
| `ds_framework.utils` | `izquierda` |

## Column-naming conventions

`rename_column` applies these prefixes:

| Prefix | Type | `tipo` argument |
|---|---|---|
| `c_` | Numeric (discrete/continuous) | `"numerica"` |
| `v_` | Categorical | `"categorica"` |
| `d_` | Date | `"fecha"` |
| `t_` | Free text | `"texto"` |
| `g_` | Geographic | `"geografica"` |
