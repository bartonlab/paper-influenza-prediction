#!/usr/bin/env python3
"""Regenerate the readable manuscript statistics report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from influenza_figures.manuscript_stats import (  # noqa: E402
    collect_manuscript_statistics,
    format_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute manuscript statistics from canonical plotting data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "manuscript-statistics.md",
        help="Markdown report destination (default: docs/manuscript-statistics.md)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_manuscript_statistics(ROOT / "data" / "plotting")
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(format_markdown_report(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} statistics to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
