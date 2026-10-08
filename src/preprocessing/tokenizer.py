"""
ALGORITHM 1 - TOKENIZATION
--------------------------
Lower-cases the text, throws away punctuation and splits it into alphanumeric
tokens:  "Machine-Learning is fun!"  ->  ["machine", "learning", "is", "fun"]

Used for: giving the indexer and the query parser a common, clean vocabulary.
"""

import re
from typing import List

# A token is a maximal run of letters/digits (after lower-casing).
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Split ``text`` into lower-case alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())
