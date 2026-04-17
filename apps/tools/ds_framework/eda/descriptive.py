"""
Descriptive statistics, frequency tables, and data-type inspection.
"""

from __future__ import annotations

from typing import Iterable, List, Union

import numpy as np
import pandas as pd


def desc_table(
    datos: pd.DataFrame,
    vars: Union[str, Iterable[str]],
    tipo: str = "num",
) -> pd.DataFrame:
    """
    Build a descriptive-statistics table for the requested variables.

    Parameters
    ----------
    datos : pd.DataFrame
        Source data.
    vars : str or iterable of str
        Column(s) to describe.
    tipo : {'num', 'all'}, default 'num'
        - 'num': numeric-only statistics.
        - 'all': include non-numeric (uses ``include='all'`` on describe).

    Returns
    -------
    pd.DataFrame
        Summary with extended percentiles and a ``%nulos`` column.
    """
    if isinstance(vars, str):
        vars = [vars]
    vars = list(vars)

    percentiles = (0.01, 0.05, 0.25, 0.3, 0.4, 0.5, 0.75, 0.98, 0.99)

    if tipo == "num":
        x = (
            datos[vars]
            .describe(percentiles=percentiles)
            .T.reset_index()
            .fillna("")
        )
    elif tipo == "all":
        x = (
            datos[vars]
            .describe(percentiles=percentiles, include="all")
            .T.reset_index()
            .fillna("")
        )
    else:
        raise ValueError("tipo debe ser 'num' o 'all'")

    x["%nulos"] = 1 - (x["count"] / len(datos))
    return x


def freq(
    df: pd.DataFrame,
    var: Union[str, List[str]],
    display_fn=None,
) -> List[pd.DataFrame]:
    """
    Build absolute / relative / cumulative frequency tables.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    var : str or list of str
        Column(s) to tabulate.
    display_fn : callable, optional
        If provided, each frequency table is passed to ``display_fn`` (for
        notebook rendering). By default, results are simply returned.

    Returns
    -------
    list of pd.DataFrame
        One frequency table per requested variable with columns:
        ``FA`` (abs), ``FR`` (rel), ``FAA`` (cum abs), ``FRA`` (cum rel).
    """
    if not isinstance(var, list):
        var = [var]

    tables: List[pd.DataFrame] = []
    for v in var:
        aux = df[v].value_counts().to_frame().rename(columns={v: "FA"})
        # Pandas >=2.0 names the value_counts column "count"; normalize to FA.
        if "count" in aux.columns and "FA" not in aux.columns:
            aux = aux.rename(columns={"count": "FA"})
        aux["FR"] = aux["FA"] / aux["FA"].sum()
        aux[["FAA", "FRA"]] = aux[["FA", "FR"]].apply(np.cumsum)
        tables.append(aux)
        if display_fn is not None:
            display_fn(aux)

    return tables


def tipo_dato(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect the distribution of Python types per column.

    Useful for detecting mixed-type columns (e.g. strings inside a
    supposedly numeric column) before modeling.

    Returns
    -------
    pd.DataFrame
        One row per column, one column per detected Python type, values are
        the proportion of rows with that type.
    """
    x = pd.DataFrame(
        [datos[c].map(type).value_counts(normalize=True) for c in datos.columns]
    ).reset_index(drop=False).fillna("")
    return x
