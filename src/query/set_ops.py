"""
ALGORITHM 5 - POSTINGS-LIST SET OPERATIONS  (with SKIP POINTERS)
----------------------------------------------------------------
All lists are sorted lists of unique doc ids.

    intersect_with_skips  - AND.  Classic two-pointer merge, O(m + n), sped up
                            with skip pointers (see src/index/skip_pointers.py).
    union                 - OR.   Two-pointer merge.
    difference            - NOT.  ``universe - docs`` (complement), two-pointer.

``stats`` (optional dict) collects a ``comparisons`` counter so the benchmark
can show how many steps the skip pointers save.
"""

from typing import Dict, List, Optional

from src.index.skip_pointers import has_skip, skip_step


def intersect_naive(a: List[int], b: List[int], stats: Optional[Dict[str, int]] = None) -> List[int]:
    """Plain two-pointer intersection (baseline for comparison)."""
    out: List[int] = []
    i = j = comparisons = 0
    while i < len(a) and j < len(b):
        comparisons += 1
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    if stats is not None:
        stats["comparisons"] = stats.get("comparisons", 0) + comparisons
    return out


def intersect_with_skips(
    a: List[int], b: List[int], stats: Optional[Dict[str, int]] = None
) -> List[int]:
    """
    AND two sorted postings lists using skip pointers.

    Walk both lists with one pointer each.  Equal ids are matches.  Otherwise
    the pointer at the smaller id must advance - and if that list has a skip
    pointer whose target is still <= the other list's current id, we jump
    straight to the target (all entries in between are smaller, so they cannot
    match) instead of stepping one entry at a time.
    """
    out: List[int] = []
    la, lb = len(a), len(b)
    step_a, step_b = skip_step(la), skip_step(lb)
    i = j = comparisons = 0

    while i < la and j < lb:
        comparisons += 1
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            if has_skip(i, la, step_a) and a[i + step_a] <= y:
                # follow the chain of skip pointers while they do not overshoot y
                while has_skip(i, la, step_a) and a[i + step_a] <= y:
                    comparisons += 1
                    i += step_a
            else:
                i += 1
        else:
            if has_skip(j, lb, step_b) and b[j + step_b] <= x:
                while has_skip(j, lb, step_b) and b[j + step_b] <= x:
                    comparisons += 1
                    j += step_b
            else:
                j += 1

    if stats is not None:
        stats["comparisons"] = stats.get("comparisons", 0) + comparisons
    return out


def union(a: List[int], b: List[int]) -> List[int]:
    """OR: merge two sorted lists, dropping duplicates."""
    out: List[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(universe: List[int], docs: List[int]) -> List[int]:
    """NOT: every doc id in ``universe`` that is not in ``docs`` (both sorted)."""
    out: List[int] = []
    j = 0
    for doc in universe:
        while j < len(docs) and docs[j] < doc:
            j += 1
        if j < len(docs) and docs[j] == doc:
            continue
        out.append(doc)
    return out
