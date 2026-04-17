"""Exploratory Data Analysis utilities."""

from .descriptive import desc_table, freq, tipo_dato
from .correlation import correlacion
from .outliers import detectar_outliers

__all__ = [
    "desc_table",
    "freq",
    "tipo_dato",
    "correlacion",
    "detectar_outliers",
]
