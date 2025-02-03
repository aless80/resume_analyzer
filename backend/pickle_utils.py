import pickle
from pathlib import Path
from typing import Any


def store_to_pickle(obj: Any, path_pickle: Path) -> None:
    """Store object to a pickle file"""
    if path_pickle.suffix != ".pkl":
        path_pickle.with_suffix(".pkl")
    with open(path_pickle, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(path_pickle: Path) -> Any:
    """Load object from pickle file"""
    if path_pickle.suffix != ".pkl":
        path_pickle = path_pickle / ".pkl"
    with open(path_pickle, "rb") as f:
        obj = pickle.load(f)

    return obj
