import unittest

from src.preprocessing.porter_stemmer import stem
from src.preprocessing.processor import TextProcessor
from src.preprocessing.tokenizer import tokenize

# Classic examples from Porter's 1980 paper
PORTER_PAIRS = {
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    "feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled",
    "motoring": "motor", "sing": "sing", "conflated": "conflat", "troubled": "troubl",
    "sized": "size", "hopping": "hop", "tanned": "tan", "falling": "fall",
    "hissing": "hiss", "fizzed": "fizz", "failing": "fail", "filing": "file",
    "happy": "happi", "sky": "sky", "relational": "relat", "conditional": "condit",
    "rational": "ration", "valenci": "valenc", "digitizer": "digit", "operator": "oper",
    "feudalism": "feudal", "decisiveness": "decis", "hopefulness": "hope",
    "formaliti": "formal", "sensitiviti": "sensit", "sensibiliti": "sensibl",
    "triplicate": "triplic", "formative": "form", "formalize": "formal",
    "electriciti": "electr", "electrical": "electr", "hopeful": "hope", "goodness": "good",
    "revival": "reviv", "allowance": "allow", "inference": "infer", "airliner": "airlin",
    "gyroscopic": "gyroscop", "adjustable": "adjust", "defensible": "defens",
    "irritant": "irrit", "replacement": "replac", "adjustment": "adjust",
    "dependent": "depend", "adoption": "adopt", "communism": "commun",
    "activate": "activ", "effective": "effect", "bowdlerize": "bowdler",
    "probate": "probat", "rate": "rate", "cease": "ceas", "controll": "control",
    "roll": "roll", "computing": "comput", "computer": "comput", "software": "softwar",
}


class TokenizerTests(unittest.TestCase):
    def test_tokenize_lowercases_and_strips_punctuation(self):
        self.assertEqual(tokenize("Machine-Learning is fun!"), ["machine", "learning", "is", "fun"])


class PorterTests(unittest.TestCase):
    def test_known_pairs(self):
        wrong = {w: (stem(w), e) for w, e in PORTER_PAIRS.items() if stem(w) != e}
        self.assertEqual(wrong, {})


class ProcessorTests(unittest.TestCase):
    def test_pipeline(self):
        stems, surface = TextProcessor().analyze("The computers are computing software")
        self.assertEqual(stems, ["comput", "comput", "softwar"])
        self.assertEqual(surface, ["computers", "computing", "software"])


if __name__ == "__main__":
    unittest.main()
