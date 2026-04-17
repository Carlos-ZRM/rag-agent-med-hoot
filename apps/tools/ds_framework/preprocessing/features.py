"""
Feature engineering helpers: train/test split, scaling, encoding,
and a one-call ColumnTransformer builder.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def split_train_test(
    data: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a frame into X_train, X_test, y_train, y_test.

    Parameters
    ----------
    data : pd.DataFrame
    target : str
        Name of the target column.
    test_size : float, default 0.2
    stratify : bool, default False
        Stratify on the target column (appropriate for classification).
    random_state : int, default 42
    """
    if target not in data.columns:
        raise ValueError(f"Target '{target}' no existe en data")

    y = data[target]
    X = data.drop(columns=[target])
    strat = y if stratify else None

    return train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=random_state
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    columns: Optional[Iterable[str]] = None,
    method: str = "standard",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], object]:
    """
    Fit a scaler on ``X_train`` and transform ``X_train`` (and optionally
    ``X_test``).

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    columns : iterable of str, optional
        Columns to scale. If None, all numeric columns are used.
    method : {'standard', 'minmax', 'robust'}

    Returns
    -------
    X_train_s, X_test_s, scaler
    """
    if method not in SCALERS:
        raise ValueError(f"method debe ser uno de {list(SCALERS)}")

    if columns is None:
        columns = X_train.select_dtypes(include=np.number).columns.tolist()
    columns = list(columns)

    scaler = SCALERS[method]()
    X_train = X_train.copy()
    X_train[columns] = scaler.fit_transform(X_train[columns])

    if X_test is not None:
        X_test = X_test.copy()
        X_test[columns] = scaler.transform(X_test[columns])

    return X_train, X_test, scaler


def encode_categoricals(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    columns: Optional[Iterable[str]] = None,
    strategy: str = "onehot",
    handle_unknown: str = "ignore",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], object]:
    """
    Encode categorical columns.

    Parameters
    ----------
    strategy : {'onehot', 'ordinal'}
    """
    if columns is None:
        columns = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    columns = list(columns)

    if not columns:
        return X_train, X_test, None

    if strategy == "onehot":
        encoder = OneHotEncoder(
            handle_unknown=handle_unknown, sparse_output=False
        )
        fitted = encoder.fit(X_train[columns])

        def _apply(df: pd.DataFrame) -> pd.DataFrame:
            arr = fitted.transform(df[columns])
            new_cols = fitted.get_feature_names_out(columns)
            out = pd.DataFrame(arr, columns=new_cols, index=df.index)
            return pd.concat([df.drop(columns=columns), out], axis=1)

    elif strategy == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        fitted = encoder.fit(X_train[columns])

        def _apply(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df[columns] = fitted.transform(df[columns])
            return df

    else:
        raise ValueError("strategy debe ser 'onehot' u 'ordinal'")

    X_train_e = _apply(X_train)
    X_test_e = _apply(X_test) if X_test is not None else None
    return X_train_e, X_test_e, fitted


def build_preprocessor(
    numeric_cols: Iterable[str],
    categorical_cols: Iterable[str],
    numeric_scaler: str = "standard",
    categorical_strategy: str = "onehot",
    numeric_impute: str = "median",
    categorical_impute: str = "most_frequent",
) -> ColumnTransformer:
    """
    Construct a reusable :class:`~sklearn.compose.ColumnTransformer` that
    imputes, scales numeric features, and encodes categoricals.

    The returned transformer is *unfit* — wrap it in a pipeline or call
    ``.fit()`` yourself.
    """
    if numeric_scaler not in SCALERS:
        raise ValueError(f"numeric_scaler debe ser uno de {list(SCALERS)}")

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=numeric_impute)),
            ("scaler", SCALERS[numeric_scaler]()),
        ]
    )

    if categorical_strategy == "onehot":
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif categorical_strategy == "ordinal":
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    else:
        raise ValueError("categorical_strategy debe ser 'onehot' u 'ordinal'")

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=categorical_impute)),
            ("encoder", cat_encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(numeric_cols)),
            ("cat", cat_pipe, list(categorical_cols)),
        ],
        remainder="drop",
    )
