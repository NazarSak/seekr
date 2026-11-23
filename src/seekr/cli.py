"""Command line interface:  seekr index | search | serve | bench"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import persist
from .engine import SearchEngine
from .snippets import ANSI, PLAIN

DEFAULT_INDEX = Path(".seekr") / "index.json.gz"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def cmd_index(args) -> int:
    import time

    root = Path(args.corpus)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    engine, report = SearchEngine.build(root, stem=not args.no_stem, stop=not args.keep_stop_words)
    build_s = time.perf_counter() - t0
    if report.loaded == 0:
        print("error: no readable text files (.txt/.md) found", file=sys.stderr)
        return 2
    size = persist.save(engine.index, Path(args.out), root=str(root.resolve()))
    idx = engine.index
    print(f"indexed {report.loaded} documents ({_fmt_bytes(report.total_bytes)} of text)")
    print(f"  terms: {len(idx.postings):,}   avg doc length: {idx.avg_len:,.0f} tokens")
    print(f"  built in {build_s:.3f}s   index file: {args.out} ({_fmt_bytes(size)})")
    for path, why in report.skipped:
        print(f"  skipped {path}: {why}")
    return 0


def _load(args) -> SearchEngine | None:
    p = Path(args.index)
    if not p.exists():
        print(f"error: no index at {p}. Build one first:  seekr index <corpus-dir>", file=sys.stderr)
        return None
    try:
        return SearchEngine.load(p, Path(args.corpus) if getattr(args, "corpus", None) else None)
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: could not load index ({exc}). Re-run `seekr index`.", file=sys.stderr)
        return None


def cmd_search(args) -> int:
    engine = _load(args)
    if engine is None:
        return 2
    mk = ANSI if sys.stdout.isatty() and not args.plain else PLAIN
    res = engine.search(
        " ".join(args.query), limit=args.limit, ranker=args.ranker, explain=args.explain, markup=mk
    )
    if res.note and not res.hits:
        print(f"{res.note}  [{res.elapsed_ms:.2f} ms]")
        return 1
    print(f"{res.total} result(s) in {res.elapsed_ms:.2f} ms   (terms: {', '.join(res.terms)})\n")
    for rank, h in enumerate(res.hits, 1):
        print(f"{rank:>2}. {h.doc.title}   [{h.doc.path}]   score {h.score:.3f}")
        print(f"    {h.snippet}")
        if h.explanation:
            e = h.explanation
            print(f"    why ({e.ranker}):")
            for p in e.parts:
                print(f"      {p.term:<14} tf={p.tf:<3} df={p.df:<3} idf={p.idf:.3f}  -> {p.weight:.4f}")
            print(f"      length-norm={e.norm:.3f}  total={e.score:.4f}")
        print()
    return 0


def cmd_serve(args) -> int:
    from .web import serve

    engine = _load(args)
    if engine is None:
        return 2
    serve(engine, args.host, args.port)
    return 0


def cmd_bench(args) -> int:
    from .bench import run

    return run(args)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="seekr", description="A small search engine (inverted index, TF-IDF/BM25).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build and save an index")
    p.add_argument("corpus", help="folder with .txt/.md files")
    p.add_argument("--out", default=str(DEFAULT_INDEX))
    p.add_argument("--no-stem", action="store_true")
    p.add_argument("--keep-stop-words", action="store_true")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("search", help="run a query")
    p.add_argument("query", nargs="+")
    p.add_argument("--index", default=str(DEFAULT_INDEX))
    p.add_argument("--corpus", help="override the corpus folder stored in the index")
    p.add_argument("-n", "--limit", type=int, default=5)
    p.add_argument("--ranker", choices=["tfidf", "bm25"], default="tfidf")
    p.add_argument("--explain", action="store_true", help="teaching mode: show why each result scored as it did")
    p.add_argument("--plain", action="store_true", help="no ANSI colours")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("serve", help="start the web UI")
    p.add_argument("--index", default=str(DEFAULT_INDEX))
    p.add_argument("--corpus")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("bench", help="measure indexing time, query latency, and compare with brute-force grep")
    p.add_argument("corpus", nargs="?", help="corpus folder (default: use --synthetic)")
    p.add_argument("--synthetic", type=int, metavar="N", help="generate N synthetic documents in a temp dir")
    p.add_argument("--repeat", type=int, default=200)
    p.set_defaults(fn=cmd_bench)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
