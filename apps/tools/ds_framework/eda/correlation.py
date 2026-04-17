"""
Correlation utilities.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def correlacion(
    datos: pd.DataFrame,
    variables: Iterable[str],
    method: str = "pearson",
    flag_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute the correlation matrix of ``variables``.

    Parameters
    ----------
    datos : pd.DataFrame
    variables : iterable of str
        Columns to include.
    method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
    flag_threshold : float, optional
        If provided, adds companion ``<var>_flag`` columns marking highly
        correlated pairs (|corr| >= threshold) with ``-CORR-`` for quick
        visual inspection.

    Returns
    -------
    pd.DataFrame
        Correlation matrix with a leading ``variable`` column.
    """
    variables = list(variables)
    corr = datos[variables].corr(method=method).reset_index().rename(
        columns={"index": "variable"}
    )

    if flag_threshold is not None:
        t = abs(flag_threshold)
        for v in variables:
            corr[f"{v}_flag"] = np.where(
                corr["variable"] == v,
                "||||||||||",
                np.where(corr[v].abs() >= t, "-CORR-", "NO"),
            )

    return corr
