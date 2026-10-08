"""
ALGORITHM 8 - LEVENSHTEIN EDIT DISTANCE  (spelling suggestions)
---------------------------------------------------------------
Edit distance = minimum number of single-character insertions, deletions and
substitutions needed to turn one word into another  ("softwre" -> "software" = 1).

Dynamic programming over a matrix D[i][j] = distance between the first i
characters of ``a`` and the first j characters of ``b``:

    D[i][j] = min( D[i-1][j]   + 1,                  # delete
                   D[i][j-1]   + 1,                  # insert
                   D[i-1][j-1] + (a[i] != b[j]) )    # substitute / match

Used for: "did you mean ...?" when a query word is not in the index.

Optimisation - candidate pruning: two words can only be within distance d if
their lengths differ by at most d, so the vocabulary is bucketed by length and
only the buckets in [len-d, len+d] are compared.  The DP also stops early once
every value in a row exceeds the allowed distance.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional


def levenshtein(a: str, b: str, max_distance: Optional[int] = None) -> int:
    """
    Edit distance between ``a`` and ``b`` using two rolling DP rows.
    If ``max_distance`` is given and exceeded, returns ``max_distance + 1``.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a                       # keep ``b`` the shorter one (smaller rows)
    if max_distance is not None and len(a) - len(b) > max_distance:
        return max_distance + 1

    previous = list(range(len(b) + 1))    # D[0][j] = j
    for i, ch_a in enumerate(a, start=1):
        current = [i]                     # D[i][0] = i
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + cost,   # substitution / match
                )
            )
        if max_distance is not None and min(current) > max_distance:
            return max_distance + 1       # early exit: cannot get back under the limit
        previous = current
    return previous[-1]


class SpellChecker:
    """Suggests the closest in-vocabulary word for a misspelled one."""

    def __init__(self, word_counts: Dict[str, int], max_distance: int = 2):
        self.max_distance = max_distance
        self.word_counts = word_counts
        # candidate pruning structure: word length -> words of that length
        self._by_length: Dict[int, List[str]] = defaultdict(list)
        for word in word_counts:
            self._by_length[len(word)].append(word)

    def suggest(self, word: str) -> Optional[str]:
        """Best correction within ``max_distance`` edits, or None."""
        if word in self.word_counts:
            return word
        # short words get a tighter budget, otherwise everything is "close"
        limit = min(self.max_distance, 1 if len(word) <= 3 else self.max_distance)

        best, best_key = None, None
        for length in range(len(word) - limit, len(word) + limit + 1):  # pruning
            for candidate in self._by_length.get(length, ()):
                distance = levenshtein(word, candidate, limit)
                if distance > limit:
                    continue
                # smaller distance wins, then higher corpus frequency, then alphabetical
                key = (distance, -self.word_counts[candidate], candidate)
                if best_key is None or key < best_key:
                    best, best_key = candidate, key
        return best
