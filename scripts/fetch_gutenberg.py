"""Download a few public-domain books from Project Gutenberg into ./books.

Usage:  python scripts/fetch_gutenberg.py [target_dir]
Then:   seekr index books

Gutenberg texts are public domain in the US. The licence header/footer is
stripped so it does not pollute search results.
"""
import re
import sys
import urllib.request
from pathlib import Path

BOOKS = {  # Gutenberg id -> file name
    1342: "pride-and-prejudice",
    84: "frankenstein",
    11: "alice-in-wonderland",
    1661: "sherlock-holmes-adventures",
    345: "dracula",
    2701: "moby-dick",
    174: "picture-of-dorian-gray",
    98: "tale-of-two-cities",
    1952: "yellow-wallpaper",
    43: "jekyll-and-hyde",
}


def strip_boilerplate(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.S)
    end = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text)
    return text[start.end() if start else 0 : end.start() if end else len(text)].strip()


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "books")
    target.mkdir(parents=True, exist_ok=True)
    for book_id, name in BOOKS.items():
        out = target / f"{name}.txt"
        if out.exists():
            continue
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print("fetching", url)
        raw = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", errors="replace")
        out.write_text(strip_boilerplate(raw), encoding="utf-8")


if __name__ == "__main__":
    main()
