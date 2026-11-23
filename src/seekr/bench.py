"""Benchmark: index build, load, query latency, and a brute-force baseline."""

from __future__ import annotations

import random
import re
import statistics
import tempfile
import time
from pathlib import Path

from . import persist
from .engine import SearchEngine
from .loader import discover, read_text

DEFAULT_QUERIES = ["index", "crawl budget", "canonical", '"search intent"', "link NOT spam", "schem*", "rankng~"]


def make_synthetic(n_docs: int, out: Path, seed: int = 7) -> None:
    """Zipf-distributed vocabulary: a few very common words, a long tail of rare ones."""
    rng = random.Random(seed)
    vocab = [f"w{i:05d}x" for i in range(20000)]
    weights = [1 / (i + 1) for i in range(len(vocab))]
    for i in range(n_docs):
        words = rng.choices(vocab, weights=weights, k=rng.randint(200, 400))
        (out / f"doc{i:06d}.txt").write_text(f"Synthetic document {i}\n" + " ".join(words), encoding="utf-8")


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def run(args) -> int:
    tmp = None
    if args.synthetic:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        t0 = time.perf_counter()
        make_synthetic(args.synthetic, root)
        print(f"generated {args.synthetic} synthetic docs in {time.perf_counter() - t0:.1f}s")
        queries = ["w00001x", "w00002x w00010x", "w00500x", '"w00001x w00002x"', "w01000x OR w02000x", "w00003x NOT w00001x"]
    elif args.corpus:
        root = Path(args.corpus)
        queries = DEFAULT_QUERIES
    else:
        print("give a corpus folder or --synthetic N")
        return 2

    total_bytes = sum(p.stat().st_size for p in discover(root))
    t0 = time.perf_counter()
    engine, report = SearchEngine.build(root)
    build_s = time.perf_counter() - t0
    idx_file = Path(tempfile.gettempdir()) / "seekr-bench-index.json.gz"
    persist.save(engine.index, idx_file, root=str(root))
    t0 = time.perf_counter()
    SearchEngine.load(idx_file)
    load_s = time.perf_counter() - t0

    print(f"\ncorpus: {report.loaded:,} docs, {total_bytes / 1e6:.2f} MB text, {len(engine.index.postings):,} terms")
    print(f"index build : {build_s:.3f}s   ({total_bytes / 1e6 / build_s:.1f} MB/s)")
    print(f"index file  : {idx_file.stat().st_size / 1e6:.2f} MB   reload: {load_s:.3f}s")

    print(f"\nquery latency, {args.repeat} runs each, snippets ON (reads files), top 10 (ms):")
    print(f"  {'query':<28}{'hits':>7}{'p50':>9}{'p95':>9}   {'grep-like':>10}  speedup")
    all_p95 = []
    for q in queries:
        times = []
        res = None
        for _ in range(args.repeat):
            t = time.perf_counter()
            res = engine.search(q, limit=10)
            times.append((time.perf_counter() - t) * 1000)
        p50, p95 = statistics.median(times), _pct(times, 0.95)
        all_p95.append(p95)
        # brute force: scan every file for the first plain word of the query
        word = re.sub(r'[^\w ]', "", q.split()[0]).lower()
        pat = re.compile(re.escape(word))
        t = time.perf_counter()
        for p in discover(root):
            pat.search(read_text(p).lower())
        brute = (time.perf_counter() - t) * 1000
        print(f"  {q:<28}{res.total:>7}{p50:>9.3f}{p95:>9.3f}   {brute:>9.1f}ms  x{brute / p50:,.0f}")
    print(f"\nworst p95 across queries: {max(all_p95):.3f} ms  (target: < 100 ms)")
    if tmp:
        tmp.cleanup()
    return 0
