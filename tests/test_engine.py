import os
import tempfile
import unittest

from src.app import get_engine
from src.config import load_config
from src.index.inverted_index import InvertedIndex
from src.index.storage import JsonFileStorage
from src.query.parser import QuerySyntaxError


def make_engine(**overrides):
    config = load_config(force_reload=True)
    config = {**config, "storage": {"backend": "memory", "json_path": "unused"}}
    config.update(overrides)
    return get_engine(config, rebuild=True)


def ids(response):
    return {h.doc_id for h in response.hits}


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = make_engine()
        cls.index = cls.engine.index

    def docs(self, query):
        return self.engine.search(query, top_k=100)

    def test_single_term_uses_stemming(self):
        # "computing" must match docs containing "computer"/"computation"
        self.assertTrue(self.docs("computing").total_matches >= 1)

    def test_and_or_not(self):
        both = ids(self.docs("software AND hardware"))
        sw = ids(self.docs("software"))
        hw = ids(self.docs("hardware"))
        self.assertEqual(both, sw & hw)
        self.assertEqual(ids(self.docs("software OR hardware")), sw | hw)
        self.assertEqual(ids(self.docs("software AND NOT hardware")), sw - hw)

    def test_readme_style_query(self):
        q = '("computer" AND "science") OR ("software" AND NOT "hardware")'
        result = self.docs(q)
        cs = ids(self.docs("computer AND science"))
        sw_not_hw = ids(self.docs("software AND NOT hardware"))
        self.assertEqual(ids(result), cs | sw_not_hw)

    def test_phrase_requires_adjacency(self):
        phrase = ids(self.docs('"machine" PHRASE "learning"'))
        both = ids(self.docs("machine AND learning"))
        self.assertTrue(phrase)
        self.assertTrue(phrase <= both)
        for doc_id in phrase:
            text = self.index.documents[doc_id]["text"].lower()
            self.assertIn("machine learning", text)

    def test_quoted_multiword_equals_phrase(self):
        self.assertEqual(ids(self.docs('"machine learning"')), ids(self.docs('"machine" PHRASE "learning"')))

    def test_not_a_phrase_of_wrong_order(self):
        self.assertEqual(self.docs('"learning" PHRASE "machine"').total_matches, 0)

    def test_spelling_suggestion(self):
        response = self.docs("softwre")
        self.assertEqual(response.corrections, {"softwre": "software"})
        self.assertEqual(response.total_matches, 0)   # reported, not applied

    def test_auto_correct(self):
        config = load_config(force_reload=True)
        config = {**config, "storage": {"backend": "memory", "json_path": "unused"},
                  "spelling": {"enabled": True, "max_edit_distance": 2, "auto_correct": True}}
        engine = get_engine(config, rebuild=True)
        self.assertGreater(engine.search("softwre").total_matches, 0)

    def test_ranking_order_and_scorers(self):
        for scorer in ("bm25", "tfidf"):
            hits = self.engine.search("galaxy stars", scorer=scorer).hits
            scores = [h.score for h in hits]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertTrue(hits[0].title in ("Stars and Galaxies", "Space Telescopes"))

    def test_stopword_only_operand_is_ignored_in_phrase(self):
        self.assertEqual(ids(self.docs('"machine" PHRASE "the" PHRASE "learning"')),
                         ids(self.docs('"machine" PHRASE "learning"')))

    def test_malformed_queries(self):
        for bad in ("AND", "(software", "software AND", "NOT"):
            with self.assertRaises(QuerySyntaxError, msg=bad):
                self.docs(bad)
        with self.assertRaises(QuerySyntaxError):
            self.docs("(a AND b) PHRASE c")

    def test_top_k(self):
        self.assertLessEqual(len(self.engine.search("the OR a OR software", top_k=3).hits), 3)


class PersistenceTests(unittest.TestCase):
    def test_json_roundtrip_gives_identical_results(self):
        engine = make_engine()
        with tempfile.TemporaryDirectory() as folder:
            storage = JsonFileStorage(os.path.join(folder, "index.json"))
            storage.save(engine.index.to_dict())
            restored = InvertedIndex.from_dict(storage.load())
        self.assertEqual(restored.postings, engine.index.postings)
        self.assertEqual(restored.doc_lengths, engine.index.doc_lengths)
        self.assertEqual(restored.surface_words, engine.index.surface_words)


if __name__ == "__main__":
    unittest.main()
