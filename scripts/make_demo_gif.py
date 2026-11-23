"""Render docs/demo.gif from the REAL output of the demo commands.

Run from the repo root:  python scripts/make_demo_gif.py
Requires Pillow (pip install pillow). The commands are executed for real; only the
"typing" animation is simulated. For a true screen capture use asciinema/ScreenToGif.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo.gif"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"  # change on Windows: C:/Windows/Fonts/consola.ttf

COLS, ROWS, SIZE = 104, 25, 14
BG, FG, PROMPT, DIM = (24, 26, 31), (220, 223, 228), (120, 220, 130), (130, 136, 148)

DEMO = [
    'seekr index corpus',
    'seekr search "crawl budget" -n 1 --explain',
    'seekr search "search intent" -n 1',
    'seekr search "crawl NOT budget" -n 1',
    'seekr search "canonicl~" -n 1',
    'seekr search "the of and"',
    'seekr bench corpus --repeat 100',
]


def run(cmd: str) -> list[str]:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    argv = [sys.executable, "-m", "seekr", *shlex.split(cmd)[1:]]
    if argv[3] == "search":
        argv.append("--plain")
    p = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    lines: list[str] = []
    for raw in (p.stdout + p.stderr).splitlines():
        lines += textwrap.wrap(raw, COLS - 2, subsequent_indent="    ", drop_whitespace=False) or [""]
    return lines


def main() -> None:
    font = ImageFont.truetype(FONT, SIZE)
    cw, lh = int(font.getlength("M")), SIZE + 5
    w, h = cw * COLS + 24, lh * ROWS + 24

    def frame(lines: list[tuple[str, tuple]]) -> Image.Image:
        im = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(im)
        for i, (text, color) in enumerate(lines[-ROWS:]):
            d.text((12, 12 + i * lh), text, font=font, fill=color)
        return im

    frames, durations = [], []
    for cmd in DEMO:
        out = run(cmd)[: ROWS - 2]
        shown: list[tuple[str, tuple]] = []
        for n in range(0, len(cmd) + 1, 3):  # typing animation
            frames.append(frame([("$ " + cmd[:n] + "_", PROMPT)]))
            durations.append(40)
        shown = [("$ " + cmd, PROMPT)] + [(ln, FG) for ln in out]
        frames.append(frame(shown))
        durations.append(2600 if len(out) > 6 else 1700)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(frames)} frames, {sum(durations) / 1000:.0f}s)")


if __name__ == "__main__":
    main()
