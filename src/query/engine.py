"""
QUERY ENGINE - ties every algorithm together.

    query string
      -> lexer                        (query/lexer.py)
      -> spelling correction          (ALGORITHM 8: Levenshtein, query/spell.py)
      -> Shunting-Yard -> postfix     (ALGORITHM 7, query/parser.py)
      -> stack evaluation             (ALGORITHM 5/6: set ops + phrase matching)
      -> BM25 / TF-IDF ranking        (ALGORITHM 9, ranking/scorers.py)
      -> ranked hits
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.index.inverted_index import InvertedIndex
from src.preprocessing.processor import TextProcessor
from src.preprocessing.tokenizer import tokenize
from src.query.lexer import AND, NOT, OR, PHRASE, TERM, Token, lex
from src.query.parser import QuerySyntaxError, to_postfix
from src.query.phrase import Positions, positional_intersect
from src.query.set_ops import difference, intersect_with_skips, union
from src.query.spell import SpellChecker
from src.ranking.scorers import create_scorer


@dataclass
class Value:
    """An intermediate result on the evaluation stack."""

    docs: List[int]                              # sorted matching doc ids
    positions: Optional[Positions] = None        # kept only for plain terms / phrases
    terms: Set[str] = field(default_factory=set)  # positive terms (used for scoring)
    is_stopword: bool = False                    # operand that vanished in preprocessing


@dataclass
class Hit:
    doc_id: int
    score: float
    title: str
    snippet: str


@dataclass
class SearchResponse:
    query: str
    postfix: str
    hits: List[Hit]
    total_matches: int
    corrections: Dict[str, str]
    elapsed_ms: float


class SearchEngine:
    def __init__(self, index: InvertedIndex, processor: TextProcessor, config: dict):
        self.index = index
        self.processor = processor
        self.config = config
        spelling = config["spelling"]
        self.spell_enabled = spelling["enabled"]
        self.auto_correct = spelling["auto_correct"]
        self.spell = SpellChecker(index.surface_words, spelling["max_edit_distance"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: Optional[int] = None, scorer: Optional[str] = None) -> SearchResponse:
        started = time.perf_counter()
        top_k = top_k or self.config["ranking"]["top_k"]

        tokens = lex(query)
        tokens, corrections = self._correct_spelling(tokens)       # ALGORITHM 8
        postfix = to_postfix(tokens)                               # ALGORITHM 7
        result = self._evaluate(postfix)                           # ALGORITHM 5 + 6

        # ALGORITHM 9: rank the Boolean result set
        ranker = create_scorer(self.config, scorer)
        scored = [(ranker.score(self.index, d, result.terms), d) for d in result.docs]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))

        hits = [self._make_hit(doc_id, score) for score, doc_id in scored[:top_k]]
        return SearchResponse(
            query=query,
            postfix=" ".join(repr(t) for t in postfix),
            hits=hits,
            total_matches=len(result.docs),
            corrections=corrections,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    # ------------------------------------------------------------------
    # Spelling correction (ALGORITHM 8 - Levenshtein)
    # ------------------------------------------------------------------
    def _correct_spelling(self, tokens: List[Token]):
        """
        For each query word that is unknown to the index, find the closest
        vocabulary word.  Suggestions are always reported; they are applied to
        the query only when ``spelling.auto_correct`` is true.
        """
        corrections: Dict[str, str] = {}
        if not self.spell_enabled:
            return tokens, corrections

        fixed: List[Token] = []
        for token in tokens:
            if token.kind != TERM:
                fixed.append(token)
                continue
            words = []
            for word in tokenize(token.value):
                known = self.processor.is_stopword(word) or self.index.has_term(
                    self.processor.stem_word(word)
                )
                if not known:
                    suggestion = self.spell.suggest(word)          # <-- Levenshtein lookup
                    if suggestion and suggestion != word:
                        corrections[word] = suggestion
                        word = suggestion if self.auto_correct else word
                words.append(word)
            fixed.append(Token(TERM, " ".join(words)) if self.auto_correct else token)
        return fixed, corrections

    # ------------------------------------------------------------------
    # Postfix evaluation (ALGORITHM 5 + 6)
    # ------------------------------------------------------------------
    def _evaluate(self, postfix: List[Token]) -> Value:
        """Evaluate the RPN query with a stack of intermediate results."""
        stack: List[Value] = []

        def pop() -> Value:
            if not stack:
                raise QuerySyntaxError("Malformed query: an operator is missing an operand")
            return stack.pop()

        for token in postfix:
            if token.kind == TERM:
                stack.append(self._term_value(token.value))

            elif token.kind == NOT:                      # complement against all docs
                operand = pop()
                stack.append(Value(difference(self.index.all_doc_ids(), operand.docs)))

            elif token.kind == AND:                      # intersection with skip pointers
                right, left = pop(), pop()
                stack.append(
                    Value(intersect_with_skips(left.docs, right.docs), terms=left.terms | right.terms)
                )

            elif token.kind == OR:                       # union
                right, left = pop(), pop()
                stack.append(Value(union(left.docs, right.docs), terms=left.terms | right.terms))

            elif token.kind == PHRASE:                   # positional intersection
                right, left = pop(), pop()
                stack.append(self._phrase_value(left, right))

        if len(stack) != 1:
            raise QuerySyntaxError("Malformed query: could not reduce it to a single result")
        return stack[0]

    def _single_term(self, stem: str) -> Value:
        return Value(
            docs=self.index.sorted_docs(stem),
            positions=self.index.positions(stem),
            terms={stem},
        )

    def _term_value(self, text: str) -> Value:
        """Look up a TERM; a multi-word quoted string is evaluated as a phrase."""
        stems = self.processor.process(text)
        if not stems:                                    # e.g. "the": matches everything
            return Value(self.index.all_doc_ids(), is_stopword=True)
        value = self._single_term(stems[0])
        for stem in stems[1:]:
            value = self._phrase_value(value, self._single_term(stem))
        return value

    def _phrase_value(self, left: Value, right: Value) -> Value:
        if left.is_stopword and right.is_stopword:
            return left
        if left.is_stopword:                             # stopwords are not indexed: ignore them
            return right
        if right.is_stopword:
            return left
        if left.positions is None or right.positions is None:
            raise QuerySyntaxError("PHRASE can only be applied to plain terms or other phrases")
        positions = positional_intersect(left.positions, right.positions)
        return Value(sorted(positions), positions, left.terms | right.terms)

    # ------------------------------------------------------------------
    def _make_hit(self, doc_id: int, score: float) -> Hit:
        doc = self.index.documents[doc_id]
        text = doc["text"]
        snippet = text if len(text) <= 160 else text[:157].rstrip() + "..."
        return Hit(doc_id, round(score, 4), doc["title"], snippet)
