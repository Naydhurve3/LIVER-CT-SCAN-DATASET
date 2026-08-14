"""Validate a notebook: nbformat-valid and every code cell parses (AST).

Usage (CLI):      python validate_notebook.py <path-to-ipynb>
Usage (notebook): set NOTEBOOK_PATH below, then run the cell.
"""
import ast
import sys
from pathlib import Path

import nbformat

NOTEBOOK_PATH = None  # set this when running inside a notebook


def validate(notebook_path: Path) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            ast.parse(cell.source)
        except SyntaxError as exc:
            raise SystemExit(
                f"Syntax error in cell {i} of {notebook_path}: {exc}"
            ) from exc
    print(f"OK {notebook_path.name}: {len(nb.cells)} cells, all parse.")


if __name__ == "__main__":
    target = NOTEBOOK_PATH or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not target:
        raise SystemExit("Pass a notebook path as argv[1] or set NOTEBOOK_PATH.")
    validate(Path(target))