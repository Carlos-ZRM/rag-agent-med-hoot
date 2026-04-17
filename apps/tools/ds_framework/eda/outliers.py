"""
Outlier detection: univariate (percentiles / IQR / robust z-score) and
multivariate (Isolation Forest).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detectar_outliers(
    datos: pd.DataFrame,
    variables: List[str],
    medida_iqr: float = 1.5,
    pct_extremos: Tuple[float, float] = (0.005, 0.995),
    usar_z_robusto: bool = True,
    thr_z_robusto: float = 3.5,
    usar_isoforest: bool = True,
    contamination: Union[str, float] = "auto",
    n_estimators: int = 300,
    random_state: int = 42,
    imputar: Optional[str] = "median",
    retornar_scores: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Identify outliers using univariate and multivariate techniques.

    Univariate:
        - Extreme percentiles (``pct_extremos``)
        - Tukey IQR (``medida_iqr``)
        - (Optional) Robust z-score based on MAD (``thr_z_robusto``)

    Multivariate:
        - (Optional) Isolation Forest

    Returns
    -------
    datos_out : pd.DataFrame
        Input data with boolean/int flag columns:
        ``out_pct_extremo``, ``out_iqr``, ``out_zrob``, ``out_uni``,
        ``out_iforest``, ``out_total``.
    resumen : pd.DataFrame
        Counts and percentages per method.
    scores_iforest : pd.DataFrame, optional
        Decision function / anomaly scores from Isolation Forest (if
        ``usar_isoforest`` and ``retornar_scores`` are True).
    """
    df = datos.copy()

    faltantes = [c for c in variables if c not in df.columns]
    if faltantes:
        raise ValueError(f"Estas variables no existen en datos: {faltantes}")

    X = df[variables].copy()

    if imputar is not None:
        if imputar == "median":
            X = X.fillna(X.median(numeric_only=True))
        elif imputar == "mean":
            X = X.fillna(X.mean(numeric_only=True))
        else:
            raise ValueError("imputar debe ser 'median', 'mean' o None")

    # -- 1) UNIVARIATE --------------------------------------------------
    p_lo, p_hi = pct_extremos
    q_lo = X.quantile(p_lo)
    q_hi = X.quantile(p_hi)
    out_pct = pd.DataFrame(False, index=df.index, columns=variables)
    for v in variables:
        out_pct[v] = (X[v] < q_lo[v]) | (X[v] > q_hi[v])
    df["out_pct_extremo"] = out_pct.any(axis=1).astype(int)

    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1
    li = q1 - medida_iqr * iqr
    ls = q3 + medida_iqr * iqr
    out_iqr = pd.DataFrame(False, index=df.index, columns=variables)
    for v in variables:
        out_iqr[v] = (X[v] < li[v]) | (X[v] > ls[v])
    df["out_iqr"] = out_iqr.any(axis=1).astype(int)

    if usar_z_robusto:
        med = X.median()
        mad = (X.sub(med)).abs().median()
        mad = mad.replace(0, np.nan)
        zrob = 0.6745 * (X.sub(med)).div(mad)
        out_zrob = zrob.abs().gt(thr_z_robusto)
        df["out_zrob"] = out_zrob.any(axis=1).astype(int)
    else:
        df["out_zrob"] = 0

    df["out_uni"] = (
        (df["out_pct_extremo"] == 1)
        | (df["out_iqr"] == 1)
        | (df["out_zrob"] == 1)
    ).astype(int)

    # -- 2) ISOLATION FOREST -------------------------------------------
    scores_iforest = None
    if usar_isoforest:
        X_num = X.astype(float)
        iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        pred = iso.fit_predict(X_num)  # -1 outlier, 1 inlier
        df["out_iforest"] = (pred == -1).astype(int)

        if retornar_scores:
            scores_iforest = pd.DataFrame(
                {
                    "iforest_decision_function": iso.decision_function(X_num),
                    "iforest_score_samples": iso.score_samples(X_num),
                    "out_iforest": df["out_iforest"].values,
                },
                index=df.index,
            )
    else:
        df["out_iforest"] = 0

    df["out_total"] = (
        (df["out_uni"] == 1) | (df["out_iforest"] == 1)
    ).astype(int)

    # -- RESUMEN -------------------------------------------------------
    def _res(col: str) -> pd.Series:
        c = int(df[col].sum())
        return pd.Series({"conteo": c, "porcentaje": c / len(df)})

    resumen = pd.concat(
        [
            _res("out_pct_extremo").rename("pct_extremo"),
            _res("out_iqr").rename("iqr"),
            _res("out_zrob").rename("z_robusto"),
            _res("out_uni").rename("univariado_union"),
            _res("out_iforest").rename("isolation_forest"),
            _res("out_total").rename("total_union"),
        ],
        axis=1,
    ).T

    return df, resumen, scores_iforest
