"""SearchEngine: ties index + query + ranker + snippets together."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import persist
from .index import Doc, InvertedIndex
from .loader import load_corpus
from .query import QueryError, evaluate, parse
from .ranker import Explanation, make_ranker
from .snippets import PLAIN, Markup, build_snippet


@dataclass
class Hit:
    doc: Doc
    score: float
    snippet: str = ""
    explanation: Explanation | None = None


@dataclass
class Result:
    query: str
    hits: list[Hit]
    total: int  # matches before the `limit` cut
    elapsed_ms: float
    terms: list[str] = field(default_factory=list)
    note: str = ""  # human-readable remark (no results, stop-word-only, parse error ...)


class SearchEngine:
    def __init__(self, index: InvertedIndex, root: Path) -> None:
        self.index = index
        self.root = Path(root)

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, root: Path, *, stem: bool = True, stop: bool = True):
        idx = InvertedIndex(stem=stem, stop=stop)
        loader = load_corpus(root)
        for rel, title, text in loader:
            idx.add(rel, title, text)
        return cls(idx, root), loader.report

    @classmethod
    def load(cls, index_path: Path, root_override: Path | None = None) -> SearchEngine:
        idx, root = persist.load(index_path)
        return cls(idx, root_override or Path(root))

    # -- searching --------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        ranker: str = "tfidf",
        snippets: bool = True,
        explain: bool = False,
        markup: Markup = PLAIN,
    ) -> Result:
        t0 = time.perf_counter()
        try:
            ev = evaluate(parse(query), self.index)
        except QueryError as exc:
            return Result(query, [], 0, _ms(t0), note=f"query error: {exc}")

        if ev.docs is None:
            return Result(
                query, [], 0, _ms(t0),
                note="the query has only stop words (too common to search for) - try more specific words",
            )
        if not ev.docs:
            return Result(query, [], 0, _ms(t0), terms=ev.terms, note="no results")

        rk = make_ranker(ranker, self.index)
        scored = sorted(
            ((rk.score(d, ev.terms), d) for d in ev.docs),
            key=lambda x: (-x[0], self.index.docs[x[1]].path),  # tie-break: stable by path
        )
        top = scored[:limit]
        hits = []
        term_set = set(ev.terms)
        for score, d in top:
            doc = self.index.docs[d]
            hit = Hit(doc, score)
            if snippets:
                hit.snippet = self._snippet(doc, term_set, markup)
            if explain:
                hit.explanation = rk.explain(d, ev.terms)
            hits.append(hit)
        return Result(query, hits, len(scored), _ms(t0), terms=ev.terms)

    def _snippet(self, doc: Doc, terms: set[str], markup: Markup) -> str:
        from .loader import read_text

        try:
            text = read_text(self.root / doc.path)
        except OSError:
            return "(file no longer available)"
        text = re.sub(r"^#+[ \t]*", "", text, flags=re.MULTILINE)  # markdown heading marks are noise
        return build_snippet(text, terms, stem=self.index.stem, markup=markup)

    def suggest(self, prefix: str, n: int = 5) -> list[str]:
        prefix = prefix.casefold().strip()
        return self.index.with_prefix(prefix, limit=n) if prefix else []

    def read_doc(self, doc_id: int) -> str:
        from .loader import read_text

        return read_text(self.root / self.index.docs[doc_id].path)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
