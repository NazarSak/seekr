"""Saving / loading the index (gzip-compressed JSON).

JSON keeps the file human-inspectable (great for the "print one document's
index and verify by hand" debugging tip). Positions are delta-encoded, which
makes the file several times smaller.
"""

from __future__ import annotations

import gzip
import json
from itertools import accumulate
from pathlib import Path

from .index import Doc, InvertedIndex

FORMAT_VERSION = 1


def _delta(positions: list[int]) -> list[int]:
    out, prev = [], 0
    for p in positions:
        out.append(p - prev)
        prev = p
    return out


def _undelta(deltas: list[int]) -> list[int]:
    return list(accumulate(deltas))


def save(index: InvertedIndex, path: Path, *, root: str) -> int:
    """Write the index; returns file size in bytes."""
    payload = {
        "version": FORMAT_VERSION,
        "root": root,
        "stem": index.stem,
        "stop": index.stop,
        "docs": [[d.path, d.title, d.length] for d in index.docs],
        "postings": {
            term: {str(doc_id): _delta(pos) for doc_id, pos in docs.items()}
            for term, docs in index.postings.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)  # atomic: a crash never leaves a half-written index
    return path.stat().st_size


def load(path: Path) -> tuple[InvertedIndex, str]:
    """Return (index, corpus_root)."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"index format {payload.get('version')} is not supported (need {FORMAT_VERSION}); re-run `seekr index`"
        )
    idx = InvertedIndex(stem=payload["stem"], stop=payload["stop"])
    idx.docs = [Doc(i, p, t, n) for i, (p, t, n) in enumerate(payload["docs"])]
    idx.postings = {
        term: {int(d): _undelta(pos) for d, pos in docs.items()}
        for term, docs in payload["postings"].items()
    }
    return idx, payload["root"]
