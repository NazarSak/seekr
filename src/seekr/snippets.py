"""Snippets: a short window of the ORIGINAL text with matches highlighted."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass

from .tokenizer import WORD_RE, normalize


@dataclass(frozen=True)
class Markup:
    open: str
    close: str
    escape: Callable[[str], str] = lambda s: s


PLAIN = Markup("**", "**")
ANSI = Markup("\x1b[1;30;43m", "\x1b[0m")
HTML = Markup("<b>", "</b>", html.escape)

_WS = re.compile(r"\s+")


def _hits(text: str, terms: set[str], stem: bool) -> list[tuple[int, int]]:
    return [
        (m.start(), m.end())
        for m in WORD_RE.finditer(text)
        if normalize(m.group(), stem) in terms
    ]


def _highlight(text: str, terms: set[str], stem: bool, mk: Markup) -> str:
    out, last = [], 0
    for s, e in _hits(text, terms, stem):
        out.append(mk.escape(text[last:s]))
        out.append(mk.open + mk.escape(text[s:e]) + mk.close)
        last = e
    out.append(mk.escape(text[last:]))
    return "".join(out)


def build_snippet(
    text: str,
    terms: set[str],
    *,
    stem: bool = True,
    width: int = 160,
    markup: Markup = PLAIN,
) -> str:
    """Pick the window of ~`width` chars holding the most query-term hits.

    Boundaries snap to whitespace so words are never cut in half.
    """
    hits = _hits(text, terms, stem)
    if not hits:
        start = 0
    else:
        # anchor a window at each hit, keep the densest one
        best_start, best_count = hits[0][0], -1
        for s, _ in hits:
            lo = max(0, s - width // 5)
            count = sum(1 for hs, he in hits if hs >= lo and he <= lo + width)
            if count > best_count:
                best_start, best_count = lo, count
        start = best_start
    end = min(len(text), start + width)

    # snap to word boundaries
    if start > 0:
        nxt = text.find(" ", start)
        prv = text.rfind(" ", 0, start)
        start = prv + 1 if prv != -1 and start - prv < 20 else (nxt + 1 if nxt != -1 else start)
    if end < len(text):
        cut = text.rfind(" ", start, end)
        if cut > start:
            end = cut
    window = _WS.sub(" ", text[start:end]).strip()
    body = _highlight(window, terms, stem, markup)
    return ("… " if start > 0 else "") + body + (" …" if end < len(text) else "")
