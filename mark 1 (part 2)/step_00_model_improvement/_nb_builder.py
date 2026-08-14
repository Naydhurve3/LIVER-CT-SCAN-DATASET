"""Shared notebook-building helpers for the Step 00 model-improvement phases."""
from pathlib import Path
import textwrap

import nbformat as nbf


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def new_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ds_gpu)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return nb


def write_notebook(nb, long_path: Path, short_path: Path | None = None):
    long_path.write_text(nbf.writes(nb), encoding="utf-8")
    written = [long_path]
    if short_path is not None:
        short_path.write_text(nbf.writes(nb), encoding="utf-8")
        written.append(short_path)
    return written