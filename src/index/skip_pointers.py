"""
SKIP-POINTER LAYOUT (helper for ALGORITHM 5)
--------------------------------------------
A skip pointer is a shortcut stored inside a sorted postings list:  the entry
at index ``i`` points to the entry at index ``i + step``.  While intersecting
two lists we can follow a pointer instead of stepping through every entry in
between, whenever the target is still <= the doc id we are looking for.

The classic heuristic is a step of about sqrt(n) for a list of length n.
Pointers are *virtual*: they are fully described by (n, step), so nothing
extra has to be stored or persisted.  ``build_skip_pointers`` materialises
them for the API / visualiser.
"""

import math
from typing import Dict


def skip_step(n: int) -> int:
    """Skip distance for a list of ``n`` entries (0 = list too short for skips)."""
    return int(math.sqrt(n)) if n >= 4 else 0


def has_skip(i: int, n: int, step: int) -> bool:
    """True if index ``i`` of a list of length ``n`` carries a skip pointer."""
    return step > 0 and i % step == 0 and i + step < n


def build_skip_pointers(n: int) -> Dict[int, int]:
    """Return {from_index: to_index} for every skip pointer of an ``n``-entry list."""
    step = skip_step(n)
    return {i: i + step for i in range(0, n, step) if i + step < n} if step else {}
