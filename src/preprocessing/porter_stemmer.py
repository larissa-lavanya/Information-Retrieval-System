"""
ALGORITHM 3 - PORTER STEMMING  (M. F. Porter, 1980)
---------------------------------------------------
Reduces a word to a root form ("stem") with ordered, rule-based suffix
stripping:  computing / computed / computer  ->  comput,  software -> softwar.

Used for: matching different surface forms of the same word, which raises
recall.  Stems need not be real words - documents and queries are stemmed
identically, so they still match.

Key idea - the *measure* m of a word: if a word is written as
[C](VC)^m[V] (C = run of consonants, V = run of vowels), m is the number of
VC pairs.  Many rules only fire when the remaining stem is "long enough"
(m > 0 or m > 1), which stops short words being over-stripped.

The algorithm runs 5 steps (1a, 1b, 1c, 2, 3, 4, 5a, 5b) in order.
"""

import re
from typing import List, Tuple

_VOWELS = "aeiou"


# --------------------------------------------------------------------------
# Helper predicates
# --------------------------------------------------------------------------
def _is_consonant(word: str, i: int) -> bool:
    """True if ``word[i]`` is a consonant ('y' is a consonant unless it follows one)."""
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return True if i == 0 else not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's measure m: the number of vowel-run -> consonant-run transitions."""
    if not stem:
        return 0
    form = "".join("c" if _is_consonant(stem, i) else "v" for i in range(len(stem)))
    collapsed = re.sub(r"(.)\1+", r"\1", form)  # "ccvvcc" -> "cvc"
    return collapsed.count("vc")


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """Consonant-vowel-consonant ending where the last consonant is not w, x or y."""
    n = len(word)
    if n < 3:
        return False
    return (
        _is_consonant(word, n - 3)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


def _apply_rules(word: str, rules: List[Tuple[str, str]], min_measure: int) -> str:
    """
    Apply the *longest* matching suffix rule.  The replacement is made only if
    the remaining stem has measure > ``min_measure``; if the longest suffix
    matches but the condition fails, no other rule is tried (as in Porter's paper).
    """
    for suffix, replacement in sorted(rules, key=lambda r: -len(r[0])):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            return stem + replacement if _measure(stem) > min_measure else word
    return word


# --------------------------------------------------------------------------
# Rule tables
# --------------------------------------------------------------------------
_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]
_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]
_STEP4 = [
    ("al", ""), ("ance", ""), ("ence", ""), ("er", ""), ("ic", ""), ("able", ""),
    ("ible", ""), ("ant", ""), ("ement", ""), ("ment", ""), ("ent", ""),
    ("ion", ""), ("ou", ""), ("ism", ""), ("ate", ""), ("iti", ""), ("ous", ""),
    ("ive", ""), ("ize", ""),
]


# --------------------------------------------------------------------------
# The steps
# --------------------------------------------------------------------------
def _step1a(word: str) -> str:
    """Plurals: sses->ss, ies->i, ss->ss, s->''."""
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    """-eed / -ed / -ing, with clean-up of the resulting stem."""
    if word.endswith("eed"):
        return word[:-1] if _measure(word[:-3]) > 0 else word

    stem = None
    if word.endswith("ed"):
        stem = word[:-2]
    elif word.endswith("ing"):
        stem = word[:-3]
    if stem is None or not _contains_vowel(stem):
        return word

    word = stem
    if word.endswith(("at", "bl", "iz")):
        return word + "e"                       # conflat(ed) -> conflate
    if _ends_double_consonant(word) and word[-1] not in "lsz":
        return word[:-1]                        # hopp(ing)  -> hop
    if _measure(word) == 1 and _ends_cvc(word):
        return word + "e"                       # fil(ing)   -> file
    return word


def _step1c(word: str) -> str:
    """Terminal y -> i when the stem contains a vowel (happy -> happi)."""
    if word.endswith("y") and _contains_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _step2(word: str) -> str:
    return _apply_rules(word, _STEP2, 0)


def _step3(word: str) -> str:
    return _apply_rules(word, _STEP3, 0)


def _step4(word: str) -> str:
    """Remove -ant, -ence, ... when m > 1.  '-ion' additionally needs a stem ending in s/t."""
    if word.endswith("ion"):
        stem = word[:-3]
        return stem if stem.endswith(("s", "t")) and _measure(stem) > 1 else word
    return _apply_rules(word, [r for r in _STEP4 if r[0] != "ion"], 1)


def _step5a(word: str) -> str:
    """Remove a final -e when m > 1, or m == 1 and the stem is not CVC."""
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            return stem
    return word


def _step5b(word: str) -> str:
    """-ll -> -l when m > 1 (controll -> control)."""
    if word.endswith("ll") and _measure(word) > 1:
        return word[:-1]
    return word


def stem(word: str) -> str:
    """Return the Porter stem of a lower-case ``word``."""
    if len(word) <= 2:
        return word
    for step in (_step1a, _step1b, _step1c, _step2, _step3, _step4, _step5a, _step5b):
        word = step(word)
    return word
