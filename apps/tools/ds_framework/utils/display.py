"""
DataFrame styling helpers for notebook rendering.
"""

from __future__ import annotations

import pandas as pd


def izquierda(df: pd.DataFrame):
    """
    Return a left-aligned :class:`~pandas.io.formats.style.Styler` for
    ``df`` (both headers and cell contents).

    Intended for Jupyter / IPython rendering. In a plain terminal the
    returned object still carries the original data via ``.data``.
    """
    styled = df.style.set_properties(**{"text-align": "left"})
    styled = styled.set_table_styles(
        [dict(selector="th", props=[("text-align", "left")])]
    )
    return styled
