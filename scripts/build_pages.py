"""Export the corpus to docs/corpus.json for the GitHub Pages demo.

Run from the repo root:  python scripts/build_pages.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
docs = []
for p in sorted((ROOT / "corpus").glob("*.md")):
    text = p.read_text(encoding="utf-8")
    title = next((ln.lstrip("#").strip() for ln in text.splitlines() if ln.strip()), p.name)
    body = "\n".join(text.splitlines()[1:]).strip()
    docs.append({"path": p.name, "title": title, "text": body})
(ROOT / "docs" / "corpus.json").write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote docs/corpus.json ({len(docs)} documents)")
