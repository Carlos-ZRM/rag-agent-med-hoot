"""
Hyperparameter tuning helpers wrapping scikit-learn's GridSearchCV and
RandomizedSearchCV.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


def grid_search(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    verbose: int = 0,
    refit: bool = True,
) -> GridSearchCV:
    """
    Run an exhaustive grid search and return the fitted ``GridSearchCV``.

    The best estimator is available at ``.best_estimator_`` and the full
    score table at ``.cv_results_``.
    """
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=refit,
    )
    gs.fit(X, y)
    return gs


def random_search(
    model: BaseEstimator,
    param_distributions: Dict[str, Any],
    X: ArrayLike,
    y: ArrayLike,
    n_iter: int = 20,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    verbose: int = 0,
    random_state: int = 42,
    refit: bool = True,
) -> RandomizedSearchCV:
    """
    Run a randomized search over ``param_distributions`` and return the
    fitted ``RandomizedSearchCV``.
    """
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        refit=refit,
    )
    rs.fit(X, y)
    return rs
