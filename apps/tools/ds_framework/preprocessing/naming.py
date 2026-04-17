"""
Column-naming convention helpers.

Prefix conventions
------------------
- ``c_`` : numeric variables (discrete and continuous)
- ``v_`` : categorical variables
- ``d_`` : date variables
- ``t_`` : text variables (comments, descriptions, URLs)
- ``g_`` : geographic variables
"""

from __future__ import annotations

import pandas as pd


NAMES = pd.DataFrame(
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


def rename_column(
    data: pd.DataFrame,
    var: str,
    tipo: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Rename a column by prefixing it with the convention for its semantic type.

    Parameters
    ----------
    data : pd.DataFrame
        Frame to mutate in-place.
    var : str
        Column name to rename.
    tipo : {'numerica', 'categorica', 'fecha', 'texto', 'geografica'}
        Semantic type; determines the prefix.
    verbose : bool, default True
        Print a status message.

    Returns
    -------
    pd.DataFrame
        Same ``data`` reference (renamed in place).
    """
    match = NAMES.loc[NAMES["val"] == tipo, "tipo"]
    if match.empty:
        raise ValueError(
            f"tipo='{tipo}' no es válido. Use uno de: {NAMES['val'].tolist()}"
        )
    typevar = match.iloc[0]

    new_name = f"{typevar}{var}"
    if new_name in data.columns:
        if verbose:
            print(f"{var}: ya tiene formato de nombre")
    else:
        data.rename(columns={var: new_name}, inplace=True)
        if verbose:
            print(f"{var} renombrada a: {new_name}")
    return data
