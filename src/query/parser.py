"""
ALGORITHM 7 - SHUNTING-YARD  (infix -> postfix / RPN)   [E. W. Dijkstra]
------------------------------------------------------------------------
Converts the user's infix query into postfix so that it can be evaluated with
a simple stack and no precedence / parenthesis logic.

    x OR y AND NOT z      ->     x y z NOT AND OR

Operator precedence (highest first):  PHRASE > NOT > AND > OR
    * PHRASE, AND, OR are binary and left-associative.
    * NOT is a unary *prefix* operator (right-associative).

Algorithm (output queue + operator stack), reading tokens left to right:
    TERM        -> append to output
    NOT         -> push (prefix operator: nothing to its left to resolve)
    binary op   -> pop operators of higher (or equal, for left-assoc)
                   precedence to the output, then push this operator
    '('         -> push
    ')'         -> pop to output until the matching '('
    end         -> pop everything that is left
"""

from typing import List

from src.query.lexer import AND, LPAREN, NOT, OR, PHRASE, RPAREN, TERM, Token, lex


class QuerySyntaxError(ValueError):
    """Raised for malformed queries (unbalanced parentheses, missing operands...)."""


PRECEDENCE = {PHRASE: 4, NOT: 3, AND: 2, OR: 1}
_BINARY = {AND, OR, PHRASE}


def to_postfix(tokens: List[Token]) -> List[Token]:
    output: List[Token] = []
    stack: List[Token] = []

    for token in tokens:
        if token.kind == TERM:
            output.append(token)

        elif token.kind == NOT:
            stack.append(token)                       # unary prefix: just push

        elif token.kind in _BINARY:
            # left-associative: pop while the stack top binds at least as tightly
            while (
                stack
                and stack[-1].kind != LPAREN
                and PRECEDENCE[stack[-1].kind] >= PRECEDENCE[token.kind]
            ):
                output.append(stack.pop())
            stack.append(token)

        elif token.kind == LPAREN:
            stack.append(token)

        elif token.kind == RPAREN:
            while stack and stack[-1].kind != LPAREN:
                output.append(stack.pop())
            if not stack:
                raise QuerySyntaxError("Unbalanced parentheses: unexpected ')'")
            stack.pop()                               # discard the '('

    while stack:
        top = stack.pop()
        if top.kind == LPAREN:
            raise QuerySyntaxError("Unbalanced parentheses: missing ')'")
        output.append(top)
    return output


def parse_query(query: str) -> List[Token]:
    """Lex the query string and convert it to postfix."""
    return to_postfix(lex(query))
