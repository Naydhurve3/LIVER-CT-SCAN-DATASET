"""Execute Mark 1 code cells sequentially in the project interpreter.

This avoids the stale Windows Jupyter launcher while preserving the notebook's
top-to-bottom state model. Scientific outputs are written by the notebook cells.
"""

from pathlib import Path
import traceback

import nbformat


NOTEBOOK = Path(__file__).with_name(
    "mark_1_probability_contrast_localization_diagnostic.ipynb"
)
LOG = Path(__file__).with_name("mark_1_execution.log")


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


LOG.write_text("", encoding="utf-8")
notebook = nbformat.read(NOTEBOOK, as_version=4)
namespace = {"__name__": "__main__"}

try:
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        log(f"START cell {index}")
        exec(compile(cell.source, f"{NOTEBOOK}:cell{index}", "exec"), namespace, namespace)
        log(f"PASS cell {index}")
except Exception:
    log(traceback.format_exc())
    raise
else:
    log("MARK_1_EXECUTION_COMPLETE")
