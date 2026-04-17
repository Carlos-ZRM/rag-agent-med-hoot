"""
High-level training, prediction, and evaluation helpers.

Wraps any estimator that follows the scikit-learn API
(``fit`` / ``predict`` / optional ``predict_proba``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


def _is_classifier(model: Any) -> bool:
    return isinstance(model, ClassifierMixin) or getattr(
        model, "_estimator_type", None
    ) == "classifier"


def _is_regressor(model: Any) -> bool:
    return isinstance(model, RegressorMixin) or getattr(
        model, "_estimator_type", None
    ) == "regressor"


def train_model(
    model: BaseEstimator,
    X_train: ArrayLike,
    y_train: ArrayLike,
    **fit_kwargs,
) -> BaseEstimator:
    """Fit ``model`` on the training data and return it."""
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def predict(
    model: BaseEstimator,
    X: ArrayLike,
    proba: bool = False,
) -> np.ndarray:
    """
    Predict labels (or probabilities with ``proba=True``).

    Raises
    ------
    AttributeError
        If ``proba=True`` and the model does not implement
        ``predict_proba``.
    """
    if proba:
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                f"{type(model).__name__} no implementa predict_proba"
            )
        return model.predict_proba(X)
    return model.predict(X)


def evaluate(
    model: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    task: Optional[str] = None,
    average: str = "weighted",
) -> Dict[str, Any]:
    """
    Compute a standard set of metrics appropriate to the task.

    Parameters
    ----------
    task : {'classification', 'regression'}, optional
        If None, inferred from the estimator type.
    average : str, default 'weighted'
        Averaging strategy for multi-class precision/recall/f1.

    Returns
    -------
    dict
        Metric name -> value (plus ``confusion_matrix`` / ``report`` for
        classification).
    """
    if task is None:
        if _is_classifier(model):
            task = "classification"
        elif _is_regressor(model):
            task = "regression"
        else:
            raise ValueError(
                "No se pudo inferir el tipo de tarea; especifique task="
                "'classification' o 'regression'"
            )

    y_pred = model.predict(X)
    metrics: Dict[str, Any] = {}

    if task == "classification":
        metrics["accuracy"] = accuracy_score(y, y_pred)
        metrics["precision"] = precision_score(
            y, y_pred, average=average, zero_division=0
        )
        metrics["recall"] = recall_score(
            y, y_pred, average=average, zero_division=0
        )
        metrics["f1"] = f1_score(y, y_pred, average=average, zero_division=0)
        metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()
        metrics["report"] = classification_report(
            y, y_pred, output_dict=True, zero_division=0
        )

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y, proba[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y, proba, multi_class="ovr", average=average
                    )
            except Exception:
                # Some datasets / targets don't support ROC-AUC; skip silently.
                pass

    elif task == "regression":
        metrics["mae"] = mean_absolute_error(y, y_pred)
        metrics["mse"] = mean_squared_error(y, y_pred)
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["r2"] = r2_score(y, y_pred)

    else:
        raise ValueError("task debe ser 'classification' o 'regression'")

    return metrics


@dataclass
class ModelTrainer:
    """
    Stateful wrapper bundling a model with its task type and last metrics.

    Example
    -------
    >>> trainer = ModelTrainer(RandomForestClassifier())
    >>> trainer.fit(X_train, y_train)
    >>> metrics = trainer.evaluate(X_test, y_test)
    """

    model: BaseEstimator
    task: Optional[str] = None
    metrics_: Dict[str, Any] = field(default_factory=dict)
    is_fitted_: bool = False

    def __post_init__(self) -> None:
        if self.task is None:
            if _is_classifier(self.model):
                self.task = "classification"
            elif _is_regressor(self.model):
                self.task = "regression"

    def fit(self, X: ArrayLike, y: ArrayLike, **fit_kwargs) -> "ModelTrainer":
        train_model(self.model, X, y, **fit_kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, X: ArrayLike, proba: bool = False) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("El modelo aún no se ha entrenado (fit)")
        return predict(self.model, X, proba=proba)

    def evaluate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        average: str = "weighted",
    ) -> Dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("El modelo aún no se ha entrenado (fit)")
        self.metrics_ = evaluate(self.model, X, y, task=self.task, average=average)
        return self.metrics_
