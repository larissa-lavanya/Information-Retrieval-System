import random
import unittest

from src.index.skip_pointers import build_skip_pointers, skip_step
from src.query.lexer import lex
from src.query.parser import QuerySyntaxError, parse_query
from src.query.phrase import positional_intersect
from src.query.set_ops import difference, intersect_naive, intersect_with_skips, union
from src.query.spell import SpellChecker, levenshtein
from src.ranking.scorers import BM25Scorer, TFIDFScorer
from src.index.inverted_index import InvertedIndex


def postfix(query):
    return " ".join(repr(t) for t in parse_query(query))


class ShuntingYardTests(unittest.TestCase):
    def test_precedence(self):
        self.assertEqual(postfix("x OR y AND NOT z"), "x y z NOT AND OR")
        self.assertEqual(postfix("a AND b OR c"), "a b AND c OR")

    def test_parentheses(self):
        self.assertEqual(postfix("a AND (b OR c)"), "a b c OR AND")

    def test_phrase_binds_tighter_than_not(self):
        self.assertEqual(postfix("NOT a PHRASE b"), "a b PHRASE NOT")

    def test_implicit_and(self):
        self.assertEqual(postfix("machine learning"), "machine learning AND")

    def test_readme_example(self):
        q = '("computer" AND "science") OR ("software" AND NOT "hardware")'
        self.assertEqual(postfix(q), "computer science AND software hardware NOT AND OR")

    def test_unbalanced(self):
        with self.assertRaises(QuerySyntaxError):
            parse_query("(a AND b")
        with self.assertRaises(QuerySyntaxError):
            parse_query("a AND b)")

    def test_quoted_phrase_is_one_term(self):
        self.assertEqual(len(lex('"machine learning"')), 1)


class SetOpsTests(unittest.TestCase):
    def test_skip_intersection_matches_naive(self):
        rng = random.Random(7)
        for _ in range(300):
            a = sorted(rng.sample(range(500), rng.randint(0, 120)))
            b = sorted(rng.sample(range(500), rng.randint(0, 120)))
            self.assertEqual(intersect_with_skips(a, b), intersect_naive(a, b))
            self.assertEqual(intersect_with_skips(a, b), sorted(set(a) & set(b)))

    def test_skips_reduce_comparisons_on_lopsided_lists(self):
        long_list = list(range(0, 10000, 2))
        short_list = [4000, 8000]
        naive, skip = {}, {}
        intersect_naive(long_list, short_list, naive)
        intersect_with_skips(long_list, short_list, skip)
        self.assertLess(skip["comparisons"], naive["comparisons"])

    def test_union_and_difference(self):
        self.assertEqual(union([1, 3, 5], [2, 3, 6]), [1, 2, 3, 5, 6])
        self.assertEqual(difference([1, 2, 3, 4, 5], [2, 4]), [1, 3, 5])

    def test_skip_pointers_layout(self):
        self.assertEqual(skip_step(3), 0)
        self.assertEqual(skip_step(16), 4)
        self.assertEqual(build_skip_pointers(16), {0: 4, 4: 8, 8: 12})


class PhraseTests(unittest.TestCase):
    def test_adjacent_positions(self):
        machine = {1: [3, 10], 2: [5]}
        learning = {1: [4, 20], 2: [9], 3: [1]}
        self.assertEqual(positional_intersect(machine, learning), {1: [4]})

    def test_chaining(self):
        a, b, c = {1: [0]}, {1: [1]}, {1: [2]}
        self.assertEqual(positional_intersect(positional_intersect(a, b), c), {1: [2]})


class LevenshteinTests(unittest.TestCase):
    def test_distances(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("softwre", "software"), 1)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("same", "same"), 0)

    def test_max_distance_cutoff(self):
        self.assertGreater(levenshtein("abcdef", "uvwxyz", max_distance=2), 2)

    def test_spellchecker(self):
        checker = SpellChecker({"software": 10, "hardware": 4, "computer": 7}, max_distance=2)
        self.assertEqual(checker.suggest("softwre"), "software")
        self.assertEqual(checker.suggest("compter"), "computer")
        self.assertIsNone(checker.suggest("zzzzzzzz"))


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.index = InvertedIndex()
        self.index.add_document(1, "a", "", ["cat", "cat", "cat", "dog"], [])
        self.index.add_document(2, "b", "", ["cat", "bird", "bird", "bird"], [])
        self.index.add_document(3, "c", "", ["bird", "fish"], [])

    def test_more_occurrences_rank_higher(self):
        for scorer in (BM25Scorer(), TFIDFScorer()):
            self.assertGreater(scorer.score(self.index, 1, ["cat"]), scorer.score(self.index, 2, ["cat"]))

    def test_non_matching_doc_scores_zero(self):
        self.assertEqual(BM25Scorer().score(self.index, 3, ["cat"]), 0.0)

    def test_bm25_saturates(self):
        idx = InvertedIndex()
        idx.add_document(1, "a", "", ["x"] * 2 + ["y"] * 8, [])
        idx.add_document(2, "b", "", ["x"] * 20 + ["y"] * 8, [])
        idx.add_document(3, "c", "", ["y"] * 10, [])
        s = BM25Scorer()
        low, high = s.score(idx, 1, ["x"]), s.score(idx, 2, ["x"])
        self.assertLess(high, low * 5)   # 10x the tf gives far less than 10x the score


if __name__ == "__main__":
    unittest.main()
