"""
Text-processing pipeline shared by the indexer (documents) and the query
engine (queries).  Using the *same* pipeline on both sides is what makes
"softwares" in a document match "software" in a query.

    raw text -> tokenize -> remove stopwords -> Porter stem

Algorithms used (each lives in its own module):
    * tokenizer.tokenize          - Algorithm 1: tokenization
    * stopwords.STOPWORDS         - Algorithm 2: stopword removal
    * porter_stemmer.stem         - Algorithm 3: Porter stemming
"""

from typing import Dict, List, Tuple

from src.preprocessing.porter_stemmer import stem
from src.preprocessing.stopwords import STOPWORDS
from src.preprocessing.tokenizer import tokenize


class TextProcessor:
    def __init__(self, remove_stopwords: bool = True, stemming: bool = True):
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self._stem_cache: Dict[str, str] = {}  # stemming is pure, so memoise it

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "TextProcessor":
        pre = config.get("preprocessing", {})
        return cls(pre.get("remove_stopwords", True), pre.get("stemming", True))

    def is_stopword(self, word: str) -> bool:
        return self.remove_stopwords and word in STOPWORDS

    def stem_word(self, word: str) -> str:
        """Stem a single already-lower-cased word (memoised)."""
        if not self.stemming:
            return word
        cached = self._stem_cache.get(word)
        if cached is None:
            cached = self._stem_cache[word] = stem(word)  # <-- Porter stemming
        return cached

    # ------------------------------------------------------------------
    def analyze(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Run the full pipeline.

        Returns two aligned lists ``(stems, surface_words)``:
            stems          - what goes into the inverted index
            surface_words  - the lower-cased original word of each stem
                             (used to build the spell-checker vocabulary)
        Positions in the index are indices into ``stems`` (i.e. counted
        *after* stopword removal).
        """
        stems: List[str] = []
        surface: List[str] = []
        for token in tokenize(text):                 # <-- tokenization
            if self.is_stopword(token):              # <-- stopword removal
                continue
            stems.append(self.stem_word(token))      # <-- Porter stemming
            surface.append(token)
        return stems, surface

    def process(self, text: str) -> List[str]:
        """Convenience wrapper returning only the stems."""
        return self.analyze(text)[0]
