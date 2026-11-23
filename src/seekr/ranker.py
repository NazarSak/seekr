"""Ranking: TF-IDF (default) and BM25, both with a per-term explanation.

TF-IDF as implemented
---------------------
    weight(t, d) = (1 + ln tf(t,d)) * idf(t)
    idf(t)       = ln(1 + N / df(t))          # smoothed: never 0, never negative
    score(d, q)  = sum_t weight(t, d) / sqrt(len(d))

* 1 + ln(tf)   : the 10th occurrence matters less than the 1st (sub-linear tf).
* idf          : rare terms say more about a document than common ones.
* / sqrt(len)  : mild length normalisation, so long books don't win by size alone.

Plain textbook idf = ln(N/df) is exactly 0 for a term present in every
document - the "zero TF-IDF" bug from the lab's troubleshooting list. The
"1 +" inside the log avoids it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .index import InvertedIndex


@dataclass
class TermPart:
    term: str
    tf: int
    df: int
    idf: float
    weight: float  # contribution to the final score


@dataclass
class Explanation:
    doc_id: int
    ranker: str
    parts: list[TermPart]
    norm: float  # length normalisation info (sqrt(len) for tfidf, bm25 length factor)
    score: float


class Ranker:
    name = "base"

    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    def explain(self, doc_id: int, terms: list[str]) -> Explanation:
        raise NotImplementedError

    def score(self, doc_id: int, terms: list[str]) -> float:
        return self.explain(doc_id, terms).score


class TfIdf(Ranker):
    name = "tfidf"

    def __init__(self, index: InvertedIndex) -> None:
        super().__init__(index)
        self._idf: dict[str, float] = {}

    def score(self, doc_id: int, terms: list[str]) -> float:
        # hot path: same maths as explain(), without building objects
        post = self.index.postings
        total = 0.0
        for t in terms:
            positions = post.get(t, {}).get(doc_id)
            if positions:
                idf = self._idf.get(t)
                if idf is None:
                    idf = self._idf[t] = self.idf(t)
                total += (1 + math.log(len(positions))) * idf
        return total / math.sqrt(max(self.index.docs[doc_id].length, 1))

    def idf(self, term: str) -> float:
        df = self.index.df(term)
        return math.log(1 + self.index.n_docs / df) if df else 0.0

    def explain(self, doc_id: int, terms: list[str]) -> Explanation:
        idx = self.index
        norm = math.sqrt(max(idx.docs[doc_id].length, 1))
        parts = []
        for t in terms:
            tf = idx.tf(t, doc_id)
            if not tf:
                continue
            idf = self.idf(t)
            parts.append(TermPart(t, tf, idx.df(t), idf, (1 + math.log(tf)) * idf / norm))
        return Explanation(doc_id, self.name, parts, norm, sum(p.weight for p in parts))


class BM25(Ranker):
    name = "bm25"
    k1 = 1.5
    b = 0.75

    def __init__(self, index: InvertedIndex) -> None:
        super().__init__(index)
        self._avg = index.avg_len or 1.0  # computed once, not per document

    def score(self, doc_id: int, terms: list[str]) -> float:
        post = self.index.postings
        avg = self._avg
        lf = 1 - self.b + self.b * self.index.docs[doc_id].length / avg
        total = 0.0
        for t in terms:
            positions = post.get(t, {}).get(doc_id)
            if positions:
                tf = len(positions)
                total += self.idf(t) * tf * (self.k1 + 1) / (tf + self.k1 * lf)
        return total

    def idf(self, term: str) -> float:
        n, df = self.index.n_docs, self.index.df(term)
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if df else 0.0

    def explain(self, doc_id: int, terms: list[str]) -> Explanation:
        idx = self.index
        avg = self._avg
        dl = idx.docs[doc_id].length
        length_factor = 1 - self.b + self.b * dl / avg
        parts = []
        for t in terms:
            tf = idx.tf(t, doc_id)
            if not tf:
                continue
            idf = self.idf(t)
            w = idf * tf * (self.k1 + 1) / (tf + self.k1 * length_factor)
            parts.append(TermPart(t, tf, idx.df(t), idf, w))
        return Explanation(doc_id, self.name, parts, length_factor, sum(p.weight for p in parts))


RANKERS = {"tfidf": TfIdf, "bm25": BM25}


def make_ranker(name: str, index: InvertedIndex) -> Ranker:
    try:
        return RANKERS[name](index)
    except KeyError:
        raise ValueError(f"unknown ranker {name!r}; choose from {', '.join(RANKERS)}") from None
