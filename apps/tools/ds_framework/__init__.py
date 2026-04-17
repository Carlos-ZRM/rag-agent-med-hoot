"""
ds_framework
============

A lightweight framework for data-science EDA, preprocessing, and
machine-learning model workflows.

Subpackages
-----------
- eda            : exploratory data analysis (descriptive stats, correlation, outliers)
- preprocessing  : column naming conventions and feature engineering helpers
- models         : training, evaluation, persistence, and hyperparameter tuning
- utils          : display and formatting helpers
"""

from . import eda
from . import preprocessing
from . import models
from . import utils

__all__ = ["eda", "preprocessing", "models", "utils"]
__version__ = "0.1.0"
