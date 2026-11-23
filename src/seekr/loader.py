"""Reading the corpus from disk, defensively."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TEXT_EXTS = {".txt", ".md", ".text"}
MAX_BYTES = 200 * 1024 * 1024  # skip absurdly large single files


@dataclass
class LoadReport:
    loaded: int = 0
    total_bytes: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)


def discover(root: Path) -> list[Path]:
    """All text files under root, sorted so indexing is deterministic."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTS)


def read_text(path: Path) -> str:
    """Read a file as text without ever crashing on encoding.

    UTF-8 (with BOM tolerated) first, then cp1252, finally UTF-8 with
    replacement characters.
    """
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:90]
    return fallback


def load_corpus(root: Path):
    """Yield (relative_posix_path, title, text); fills a LoadReport as a side effect.

    Usage:  gen = load_corpus(root); for item in gen: ...; report = gen.report
    """
    return _Loader(root)


class _Loader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.report = LoadReport()

    def __iter__(self):
        for p in discover(self.root):
            rel = p.relative_to(self.root).as_posix()
            try:
                size = p.stat().st_size
                if size == 0:
                    self.report.skipped.append((rel, "empty file"))
                    continue
                if size > MAX_BYTES:
                    self.report.skipped.append((rel, f"too large ({size} bytes)"))
                    continue
                text = read_text(p)
            except OSError as exc:
                self.report.skipped.append((rel, f"unreadable: {exc}"))
                continue
            if not text.strip():
                self.report.skipped.append((rel, "no text (whitespace only)"))
                continue
            self.report.loaded += 1
            self.report.total_bytes += size
            yield rel, title_of(text, p.name), text
