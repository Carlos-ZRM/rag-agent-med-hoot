"""Utilidades de EDA para pipelines de ciencia de datos."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def desc_table(datos: pd.DataFrame, vars: list, tipo: str) -> pd.DataFrame:
    """Estadísticos descriptivos de las variables."""
    percentiles = (0.01, 0.05, 0.25, 0.3, 0.4, 0.5, 0.75, 0.98, 0.99)
    if tipo == "num":
        x = datos[vars].describe(percentiles=percentiles).T.reset_index().fillna("")
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


def detectar_outliers(
    datos: pd.DataFrame,
    variables: list,
    medida_iqr: float = 1.5,
    pct_extremos: tuple = (0.005, 0.995),
    usar_z_robusto: bool = True,
    thr_z_robusto: float = 3.5,
    usar_isoforest: bool = True,
    contamination: float = "auto",
    n_estimators: int = 300,
    random_state: int = 42,
    imputar: str = "median",
    retornar_scores: bool = True,
):
    """
    Identifica outliers por:
      - Univariado: percentiles extremos + IQR (Tukey) + z robusto (MAD)
      - Multivariado: Isolation Forest

    Regresa:
      datos_out, resumen, scores_iforest
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

    # 1) UNIVARIADO
    # a) Percentiles extremos
    p_lo, p_hi = pct_extremos
    q_lo = X.quantile(p_lo)
    q_hi = X.quantile(p_hi)
    out_pct = pd.DataFrame(False, index=df.index, columns=variables)
    for v in variables:
        out_pct[v] = (X[v] < q_lo[v]) | (X[v] > q_hi[v])
    df["out_pct_extremo"] = out_pct.any(axis=1).astype(int)

    # b) IQR (Tukey)
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1
    li = q1 - medida_iqr * iqr
    ls = q3 + medida_iqr * iqr
    out_iqr = pd.DataFrame(False, index=df.index, columns=variables)
    for v in variables:
        out_iqr[v] = (X[v] < li[v]) | (X[v] > ls[v])
    df["out_iqr"] = out_iqr.any(axis=1).astype(int)

    # c) Z-score robusto con MAD
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

    # 2) ISOLATION FOREST
    scores_iforest = None
    if usar_isoforest:
        X_num = X.astype(float)
        iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        pred = iso.fit_predict(X_num)
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

    df["out_total"] = ((df["out_uni"] == 1) | (df["out_iforest"] == 1)).astype(int)

    # RESUMEN
    def _res(col):
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


def correlacion(datos: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Matriz de correlación."""
    corr = datos[variables].corr().reset_index().rename(columns={"index": "variable"})
    return corr


def freq(df: pd.DataFrame, var) -> pd.DataFrame:
    """Tabla de frecuencias absolutas, relativas y acumuladas."""
    if not isinstance(var, list):
        var = [var]
    results = []
    for v in var:
        aux = df[v].value_counts().to_frame().rename(columns={v: "FA"})
        aux["FR"] = aux["FA"] / aux["FA"].sum()
        aux[["FAA", "FRA"]] = aux.apply(np.cumsum)
        results.append(aux)
    return results if len(results) > 1 else results[0]


def tipo_dato(datos: pd.DataFrame) -> pd.DataFrame:
    """Proporción de tipos de dato por columna."""
    x = pd.DataFrame(
        [datos[c].map(type).value_counts(normalize=True) for c in datos.columns]
    ).reset_index(drop=False).fillna("")
    return x


def completitud(datos: pd.DataFrame) -> pd.DataFrame:
    """Proporción de nulos por columna, ordenada descendente."""
    comp = (datos.isnull().sum() / datos.shape[0]).reset_index()
    comp.columns = ["columna", "completitud"]
    return comp.sort_values(by=["completitud"], ascending=False).reset_index(drop=True)


# Convención de nombres por tipo de variable
NOMBRES_TIPO = pd.DataFrame(
    {
        "tipo": ["c_", "v_", "d_", "t_", "g_"],
        "descripcion": [
            "Variables numericas : Discretas y continuas",
            "Variables categoricas",
            "Variables tipo fecha",
            "Variables de texto : comentarios, descripciones, url",
            "Variables geograficas",
        ],
        "val": ["numerica", "categorica", "fecha", "texto", "geografica"],
    }
)


def rename_column(data: pd.DataFrame, var: str, tipo: str):
    """Renombra columna con prefijo según tipo de variable."""
    typevar = NOMBRES_TIPO.loc[NOMBRES_TIPO["val"] == tipo, "tipo"].iloc[0]
    new_name = typevar + var
    if new_name in data.columns:
        print(f"{var}: ya tiene formato de nombre")
    else:
        data.rename(columns={var: new_name}, inplace=True)
        print(f"{var} renombrada a: {new_name}")