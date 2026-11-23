"""Text analysis: tokenize -> lowercase -> (drop stop words) -> (light stem).

The same pipeline MUST be used for documents and for queries, otherwise
"Hello" in a query will never match "hello" in the index.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

# Letters/digits (any script), with inner apostrophes kept: don't, o'clock.
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)

STOP_WORDS = frozenset(
    """a an and are as at be been but by for from had has have he her his i if
    in into is it its of on or she so than that the their them then there
    these they this to was we were what when which who will with you your not
    no do does did can could would should about also""".split()
)

_VOWELS = "aeiou"


def light_stem(word: str) -> str:
    """A deliberately small suffix stripper (NOT the Porter stemmer).

    It handles the most common English plurals and verb endings so that
    "ranking", "ranked" and "ranks" meet at "rank". It is cheap and
    predictable; the price is occasional over/under-stemming (documented in
    the README).
    """
    w = word
    if len(w) <= 3 or not w.isascii():
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("xes") or w.endswith("ches") or w.endswith("shes"):
        return w[:-2]
    for suffix in ("edly", "ing", "ed"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            base = w[: -len(suffix)]
            if not any(c in _VOWELS for c in base):
                return w
            # running -> runn -> run (but keep "ll", "ss", "zz")
            if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "lsz" + _VOWELS:
                base = base[:-1]
            return base
    if w.endswith("ly") and len(w) > 5:
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def normalize(token: str, stem: bool = True) -> str:
    t = token.casefold().replace("’", "'")
    return light_stem(t) if stem else t


def analyze(
    text: str, *, stem: bool = True, stop: bool = True
) -> Iterator[tuple[str, int]]:
    """Yield (term, position) for every kept token.

    `position` counts ALL tokens, including dropped stop words. That keeps the
    distance between words honest, so the phrase "search of engines" is not
    confused with "search engines".
    """
    for pos, m in enumerate(WORD_RE.finditer(text)):
        raw = m.group().casefold().replace("’", "'")
        if stop and raw in STOP_WORDS:
            continue
        yield (light_stem(raw) if stem else raw), pos
