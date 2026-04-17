"""
Persist models (and optional metadata) to disk using joblib.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib


PathLike = Union[str, Path]


def save_model(
    model: Any,
    path: PathLike,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Path:
    """
    Persist ``model`` to ``path`` with joblib. If ``metadata`` is provided,
    a sibling JSON file (``<path>.meta.json``) is written alongside.

    Parameters
    ----------
    model : object
        Any picklable estimator.
    path : str or Path
        Target filename (``.joblib`` recommended).
    metadata : dict, optional
        Arbitrary JSON-serializable metadata (feature list, metrics,
        training dataset hash, …). A ``saved_at`` timestamp is added
        automatically.
    overwrite : bool, default True
        If False and ``path`` exists, raises ``FileExistsError``.

    Returns
    -------
    Path
        The path the model was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} ya existe y overwrite=False")

    joblib.dump(model, path)

    if metadata is not None:
        meta = dict(metadata)
        meta.setdefault("saved_at", datetime.utcnow().isoformat() + "Z")
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    return path


def load_model(
    path: PathLike,
    with_metadata: bool = False,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Load a model saved with :func:`save_model`.

    Parameters
    ----------
    with_metadata : bool, default False
        If True, returns ``(model, metadata)``. Metadata is an empty dict
        if no companion JSON file is found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    model = joblib.load(path)

    if not with_metadata:
        return model

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return model, metadata
