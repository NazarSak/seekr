"""The inverted (positional) index.

    term -> { doc_id -> [positions...] }

* len(postings[term])            = document frequency (df)
* len(postings[term][doc_id])    = term frequency in that doc (tf)
* the positions make phrase search possible

Looking a term up is one dict access: O(1) on average, independent of how many
documents exist. That is the whole trick, and why it beats grepping every file.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from .tokenizer import analyze


@dataclass(slots=True)
class Doc:
    id: int
    path: str  # path relative to the corpus root (unique identity)
    title: str
    length: int  # number of indexed tokens (stop words excluded)


class InvertedIndex:
    def __init__(self, *, stem: bool = True, stop: bool = True) -> None:
        self.stem = stem
        self.stop = stop
        self.postings: dict[str, dict[int, list[int]]] = {}
        self.docs: list[Doc] = []
        self._vocab: list[str] | None = None

    # -- building ---------------------------------------------------------
    def add(self, path: str, title: str, text: str) -> int:
        doc_id = len(self.docs)
        length = 0
        for term, pos in analyze(text, stem=self.stem, stop=self.stop):
            self.postings.setdefault(term, {}).setdefault(doc_id, []).append(pos)
            length += 1
        self.docs.append(Doc(doc_id, path, title, length))
        self._vocab = None
        return doc_id

    # -- statistics -------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return len(self.docs)

    @property
    def avg_len(self) -> float:
        return sum(d.length for d in self.docs) / len(self.docs) if self.docs else 0.0

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def tf(self, term: str, doc_id: int) -> int:
        return len(self.postings.get(term, {}).get(doc_id, ()))

    # -- vocabulary helpers (prefix / fuzzy) -------------------------------
    @property
    def vocab(self) -> list[str]:
        if self._vocab is None:
            self._vocab = sorted(self.postings)
        return self._vocab

    def with_prefix(self, prefix: str, limit: int | None = None) -> list[str]:
        v = self.vocab
        i = bisect.bisect_left(v, prefix)
        out: list[str] = []
        while i < len(v) and v[i].startswith(prefix):
            out.append(v[i])
            i += 1
        out.sort(key=lambda t: (-self.df(t), t))
        return out[:limit] if limit else out

    def within_distance(self, term: str, max_dist: int = 2) -> list[str]:
        from .fuzzy import levenshtein

        return [
            t
            for t in self.vocab
            if abs(len(t) - len(term)) <= max_dist and levenshtein(t, term, max_dist) <= max_dist
        ]
