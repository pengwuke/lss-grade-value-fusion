"""Explicit fold identities shared by artifact lookup and frozen fusion."""
from numbers import Integral


def fold_index(value):
    """Accept Python/NumPy integer indices 0..4; never truncate or coerce IDs."""
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 5:
        raise ValueError("Fold must be an integer index 0..4; boolean, floating and string values are not accepted")
    return int(value)
