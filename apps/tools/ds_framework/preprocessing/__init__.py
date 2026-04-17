"""Preprocessing utilities: column naming conventions and feature engineering."""

from .naming import rename_column, NAMES
from .features import (
    split_train_test,
    scale_features,
    encode_categoricals,
    build_preprocessor,
)

__all__ = [
    "rename_column",
    "NAMES",
    "split_train_test",
    "scale_features",
    "encode_categoricals",
    "build_preprocessor",
]
