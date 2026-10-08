"""
Query lexer: turns the query string into a flat list of tokens.

    ("computer" AND "science") OR NOT hardware
    -> ( TERM(computer) AND TERM(science) ) OR NOT TERM(hardware)

Rules
    * AND / OR / NOT / PHRASE (case-insensitive, unquoted) are operators.
    * Quoted text, or any other word, is a TERM.  A quoted multi-word string
      such as "machine learning" is a single TERM that the engine treats as a
      phrase.
    * Two adjacent operands get an implicit AND:  ``machine learning`` ==
      ``machine AND learning``.
"""

import re
from dataclasses import dataclass
from typing import List

TERM, AND, OR, NOT, PHRASE, LPAREN, RPAREN = "TERM", "AND", "OR", "NOT", "PHRASE", "(", ")"
OPERATORS = {"AND": AND, "OR": OR, "NOT": NOT, "PHRASE": PHRASE}

_LEX_RE = re.compile(r'"([^"]*)"|(\()|(\))|([^\s()"]+)')


@dataclass(frozen=True)
class Token:
    kind: str
    value: str = ""

    def __repr__(self) -> str:  # compact form used when printing postfix queries
        return f"{self.value}" if self.kind == TERM else self.kind


def lex(query: str) -> List[Token]:
    tokens: List[Token] = []
    for quoted, lparen, rparen, word in _LEX_RE.findall(query):
        if lparen:
            tokens.append(Token(LPAREN))
        elif rparen:
            tokens.append(Token(RPAREN))
        elif word:
            kind = OPERATORS.get(word.upper())
            tokens.append(Token(kind) if kind else Token(TERM, word))
        else:  # a quoted string (possibly empty)
            if quoted.strip():
                tokens.append(Token(TERM, quoted))
    return _insert_implicit_and(tokens)


def _insert_implicit_and(tokens: List[Token]) -> List[Token]:
    """Put an AND between two neighbouring operands."""
    ends_operand = {TERM, RPAREN}
    starts_operand = {TERM, NOT, LPAREN}
    out: List[Token] = []
    for token in tokens:
        if out and out[-1].kind in ends_operand and token.kind in starts_operand:
            out.append(Token(AND))
        out.append(token)
    return out
