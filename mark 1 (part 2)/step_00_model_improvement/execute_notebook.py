"""Execute a notebook with nbclient against the project venv kernel.

Usage (CLI):      python execute_notebook.py <path-to-ipynb> [timeout]
Usage (notebook): set NOTEBOOK_PATH (and optional TIMEOUT) below, then run.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = None  # set this when running inside a notebook
TIMEOUT = 3600


def run(notebook_path, timeout=3600):
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    print(f"Executed {notebook_path.name}: {len(nb.cells)} cells OK")


if __name__ == "__main__":
    target = NOTEBOOK_PATH or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not target:
        raise SystemExit("Pass a notebook path as argv[1] or set NOTEBOOK_PATH.")
    run(Path(target), timeout=int(sys.argv[2]) if len(sys.argv) > 2 else TIMEOUT)