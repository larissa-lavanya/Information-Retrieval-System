"""
ALGORITHM 6 - PHRASE MATCHING (POSITIONAL INTERSECTION)
-------------------------------------------------------
Finds documents where one term is immediately followed by another, e.g.
``"machine" PHRASE "learning"``.

Step 1: intersect the two terms' doc-id lists (using skip pointers).
Step 2: inside each shared document, keep the positions ``q`` of the right
        term for which ``q - 1`` is a position of the left term.

The result keeps the *right-hand* positions, so phrases can be chained:
``a PHRASE b PHRASE c`` becomes ``(a PHRASE b) PHRASE c``.
"""

from typing import Dict, List

from src.query.set_ops import intersect_with_skips

Positions = Dict[int, List[int]]  # doc_id -> sorted positions


def positional_intersect(left: Positions, right: Positions) -> Positions:
    result: Positions = {}
    shared_docs = intersect_with_skips(sorted(left), sorted(right))
    for doc in shared_docs:
        left_pos, right_pos = left[doc], right[doc]
        matches: List[int] = []
        i = j = 0
        while i < len(left_pos) and j < len(right_pos):
            wanted = left_pos[i] + 1                 # right term must sit right after left
            if right_pos[j] == wanted:
                matches.append(right_pos[j])
                i += 1
                j += 1
            elif right_pos[j] < wanted:
                j += 1
            else:
                i += 1
        if matches:
            result[doc] = matches
    return result
