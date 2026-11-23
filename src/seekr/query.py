"""Query language: parser + evaluator.

Syntax
------
    rank snippets            implicit AND
    rank AND snippets        explicit AND
    rank OR bm25             union
    rank NOT bm25   /  -bm25 exclusion
    "inverted index"         exact phrase (uses stored positions)
    (a OR b) c               grouping
    indx~                    fuzzy: terms within edit distance 1-2
    crawl*                   prefix expansion

Grammar (precedence: NOT > AND > OR):
    or_expr  := and_expr ( "OR" and_expr )*
    and_expr := unary ( ["AND"] unary )*
    unary    := ("NOT" | "-") unary | primary
    primary  := "(" or_expr ")" | PHRASE | WORD
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .index import InvertedIndex
from .tokenizer import analyze

_LEX = re.compile(r'"[^"]*"?|\(|\)|[^\s()"]+')


class QueryError(ValueError):
    pass


@dataclass
class Word:
    text: str


@dataclass
class Phrase:
    text: str


@dataclass
class Not:
    child: object


@dataclass
class And:
    children: list


@dataclass
class Or:
    children: list


def tokenize_query(q: str) -> list[str]:
    return _LEX.findall(q)


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.t = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self) -> str:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self):
        if not self.t:
            raise QueryError("empty query")
        node = self.or_expr()
        if self.peek() is not None:
            raise QueryError(f"unexpected {self.peek()!r}")
        return node

    def or_expr(self):
        parts = [self.and_expr()]
        while self.peek() == "OR":
            self.next()
            parts.append(self.and_expr())
        return parts[0] if len(parts) == 1 else Or(parts)

    def and_expr(self):
        parts = [self.unary()]
        while self.peek() not in (None, "OR", ")"):
            if self.peek() == "AND":
                self.next()
            parts.append(self.unary())
        return parts[0] if len(parts) == 1 else And(parts)

    def unary(self):
        tok = self.peek()
        if tok is None:
            raise QueryError("query ends unexpectedly")
        if tok == "NOT":
            self.next()
            return Not(self.unary())
        if tok.startswith("-") and len(tok) > 1 and not tok.startswith('-"'):
            self.next()
            return Not(Word(tok[1:]))
        return self.primary()

    def primary(self):
        tok = self.next()
        if tok == "(":
            node = self.or_expr()
            if self.peek() != ")":
                raise QueryError("missing closing parenthesis")
            self.next()
            return node
        if tok == ")":
            raise QueryError("unexpected ')'")
        if tok.startswith('"'):
            return Phrase(tok.strip('"'))
        if tok in ("AND", "OR"):
            raise QueryError(f"operator {tok} needs terms on both sides")
        return Word(tok)


def parse(q: str):
    return _Parser(tokenize_query(q)).parse()


@dataclass
class Evaluation:
    docs: set[int] | None  # None => query had no usable terms (e.g. only stop words)
    terms: list[str]  # index terms that contribute to ranking/highlighting (unique, ordered)


def _uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _eval_word(node: Word, idx: InvertedIndex):
    text = node.text
    fuzzy = text.endswith("~")
    prefix = text.endswith("*")
    core = text.rstrip("~*")
    pieces = list(analyze(core, stem=idx.stem, stop=idx.stop))
    if prefix and core:
        expanded = idx.with_prefix(core.casefold(), limit=200)
        if not expanded and pieces:
            expanded = [pieces[0][0]]
        return _union_of(expanded, idx)
    if not pieces:
        return None, []
    if len(pieces) > 1:  # "co-op" -> co op : behave as a phrase
        return _eval_phrase(Phrase(core), idx)
    term = pieces[0][0]
    if fuzzy:
        dist = 1 if len(term) <= 4 else 2
        return _union_of(_uniq([term, *idx.within_distance(term, dist)]), idx)
    return set(idx.postings.get(term, {})), [term]


def _union_of(terms: list[str], idx: InvertedIndex):
    docs: set[int] = set()
    for t in terms:
        docs |= set(idx.postings.get(t, {}))
    return docs, list(terms)


def _eval_phrase(node: Phrase, idx: InvertedIndex):
    pieces = list(analyze(node.text, stem=idx.stem, stop=idx.stop))
    if not pieces:
        return None, []
    terms = _uniq(t for t, _ in pieces)
    if len(pieces) == 1:
        return set(idx.postings.get(pieces[0][0], {})), terms
    first_term, first_pos = pieces[0]
    rest = [(t, p - first_pos) for t, p in pieces[1:]]
    candidates = set(idx.postings.get(first_term, {}))
    for t, _ in rest:
        candidates &= set(idx.postings.get(t, {}))
    hits: set[int] = set()
    for d in candidates:
        starts = idx.postings[first_term][d]
        others = [(set(idx.postings[t][d]), off) for t, off in rest]
        if any(all(s + off in ps for ps, off in others) for s in starts):
            hits.add(d)
    return hits, terms


def evaluate(node, idx: InvertedIndex) -> Evaluation:
    docs, terms = _eval(node, idx)
    return Evaluation(docs, _uniq(terms))


def _eval(node, idx):
    if isinstance(node, Word):
        return _eval_word(node, idx)
    if isinstance(node, Phrase):
        return _eval_phrase(node, idx)
    if isinstance(node, Not):
        docs, _ = _eval(node.child, idx)
        if docs is None:
            return None, []
        return set(range(idx.n_docs)) - docs, []
    if isinstance(node, And):
        acc: set[int] | None = None
        terms: list[str] = []
        for c in node.children:
            docs, t = _eval(c, idx)
            if docs is None:  # a stop word: no constraint
                continue
            terms += t
            acc = docs if acc is None else acc & docs
        return acc, terms
    if isinstance(node, Or):
        acc = None
        terms = []
        for c in node.children:
            docs, t = _eval(c, idx)
            if docs is None:
                continue
            terms += t
            acc = docs if acc is None else acc | docs
        return acc, terms
    raise QueryError(f"unknown node {node!r}")
