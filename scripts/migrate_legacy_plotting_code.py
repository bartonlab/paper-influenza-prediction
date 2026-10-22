#!/usr/bin/env python3
"""Move active influenza plotting code behind small compatibility wrappers.

The byte-preserving originals must already exist in the dated archive.  The
unrelated CH505/HIV/Wright-Fisher section is retained only in that archive.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


ARCHIVE_NAME = "pre_plot_reorg_2026-08-26"
UNRELATED_START = "\ndef plot_figure_ch505_structure():"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def require_archived_match(current: Path, archived: Path) -> None:
    if not current.is_file() or not archived.is_file():
        raise FileNotFoundError(f"Missing current/archive pair: {current}, {archived}")
    if digest(current) != digest(archived):
        raise RuntimeError(
            f"Current file changed after the archive snapshot; refusing to rewrite {current}"
        )


def migrate(repo: Path) -> None:
    archive = repo / "archive" / ARCHIVE_NAME / "original"
    archived_figures = archive / "figures.py"
    archived_mplot = archive / "mplot.py"
    current_figures = repo / "figures.py"
    current_mplot = repo / "mplot.py"
    require_archived_match(current_figures, archived_figures)
    require_archived_match(current_mplot, archived_mplot)

    legacy_dir = repo / "src" / "influenza_figures" / "legacy"
    if legacy_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing legacy package: {legacy_dir}")
    legacy_dir.mkdir(parents=True)

    source = archived_figures.read_text(encoding="utf-8")
    boundary = source.find(UNRELATED_START)
    if boundary < 0:
        raise RuntimeError("Could not find the audited start of unrelated plotting code")
    influenza_source = source[:boundary].rstrip() + "\n"
    expected_import = "import mplot as mp"
    if influenza_source.count(expected_import) != 1:
        raise RuntimeError("Unexpected mplot import count in archived figures.py")
    influenza_source = influenza_source.replace(
        expected_import, "from . import mplot as mp", 1
    )
    (legacy_dir / "figures.py").write_text(influenza_source, encoding="utf-8")
    (legacy_dir / "mplot.py").write_bytes(archived_mplot.read_bytes())
    (legacy_dir / "__init__.py").write_text(
        '"""Archived-compatible influenza plotting implementation.\n\n'
        "New code should use the explicit registry, styles, and canonical loaders.\n"
        '"""\n',
        encoding="utf-8",
    )
    (legacy_dir / "README.md").write_text(
        "# Legacy-compatible plotting code\n\n"
        "This package contains the influenza portion of the archived `figures.py` "
        "and the complete archived `mplot.py`. It exists so historical notebook "
        "imports continue to work while renderers move into topic modules. The "
        "unrelated CH505/HIV/Wright-Fisher tail remains only in the dated archive.\n\n"
        "Use `influenza_figures.render.legacy_style_context` around calls into this "
        "package. It applies one explicit paper, slide, poster, or grant profile to "
        "both historical modules and restores the previous settings afterward. The "
        "root `figures.py` and `mplot.py` files are compatibility aliases for older "
        "imports.\n",
        encoding="utf-8",
    )

    current_figures.write_text(
        '"""Compatibility alias for the reorganized influenza plotting package."""\n\n'
        "import importlib\n"
        "from pathlib import Path\n"
        "import sys\n\n"
        "_SRC = Path(__file__).resolve().parent / \"src\"\n"
        "if str(_SRC) not in sys.path:\n"
        "    sys.path.insert(0, str(_SRC))\n\n"
        "_IMPLEMENTATION = importlib.import_module(\"influenza_figures.legacy.figures\")\n"
        "sys.modules[__name__] = _IMPLEMENTATION\n",
        encoding="utf-8",
    )
    current_mplot.write_text(
        '"""Compatibility alias for the reorganized plotting primitives."""\n\n'
        "import importlib\n"
        "from pathlib import Path\n"
        "import sys\n\n"
        "_SRC = Path(__file__).resolve().parent / \"src\"\n"
        "if str(_SRC) not in sys.path:\n"
        "    sys.path.insert(0, str(_SRC))\n\n"
        "_IMPLEMENTATION = importlib.import_module(\"influenza_figures.legacy.mplot\")\n"
        "sys.modules[__name__] = _IMPLEMENTATION\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root",
    )
    args = parser.parse_args()
    try:
        migrate(args.repo.resolve())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
