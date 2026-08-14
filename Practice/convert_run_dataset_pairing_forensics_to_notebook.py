"""Convert the complete dataset-pairing forensic script into a Jupyter notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


PRACTICE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = PRACTICE_DIR / "run_dataset_pairing_forensics.py"
NOTEBOOK_PATH = PRACTICE_DIR / "run_dataset_pairing_forensics.ipynb"

SECTION_TITLES = {
    "1": "Source-to-derived regeneration and local offset search",
    "2": "Independent legacy-reference transform audit",
    "3": "Anatomy, body-outline, continuity, and tumor morphology audit",
    "4": "Consolidated decisions and failures",
    "5": "Evidence-focused figures and final summary",
}


def split_script(source: str) -> list[tuple[str, str]]:
    """Split the source at its numbered top-level forensic section markers."""
    lines = source.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        for number in SECTION_TITLES:
            if stripped.startswith(f"# {number}."):
                # Include the separator immediately above the numbered heading.
                start = index - 1 if index and lines[index - 1].strip().startswith("# ---") else index
                starts.append((start, number))
                break

    if len(starts) != len(SECTION_TITLES):
        found = [number for _, number in starts]
        raise RuntimeError(f"Expected sections {list(SECTION_TITLES)}, found {found}")

    parts: list[tuple[str, str]] = [("setup", "".join(lines[: starts[0][0]]).rstrip() + "\n")]
    for position, (start, number) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        parts.append((number, "".join(lines[start:end]).rstrip() + "\n"))
    return parts


def main() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    parts = split_script(source)

    cells = [
        nbformat.v4.new_markdown_cell(
            "# LiTS dataset pairing forensics — full runnable notebook\n\n"
            "This notebook is a cell-by-cell conversion of "
            "`run_dataset_pairing_forensics.py`. It performs the full source-to-derived "
            "pairing, transform, anatomy, continuity, and tumor-morphology audit.\n\n"
            "**Important:** run the cells in order. The analysis reads the staged dataset "
            "build and writes evidence files to "
            "`Practice/dataset_pairing_forensics_results/`. It does not promote or modify "
            "the canonical dataset."
        ),
        nbformat.v4.new_markdown_cell(
            "## 0. Setup, paths, helper functions, and audit-table loading\n\n"
            "Confirm `BUILD_DIR` and `LEGACY_DIR` in the next cell before running if the "
            "dataset build or legacy reference has moved."
        ),
        nbformat.v4.new_code_cell(parts[0][1]),
    ]

    for number, code in parts[1:]:
        cells.append(nbformat.v4.new_markdown_cell(f"## {number}. {SECTION_TITLES[number]}"))
        cells.append(nbformat.v4.new_code_cell(code))

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Manual review checklist\n\n"
            "After all cells finish, review the generated CSV/JSON tables and evidence "
            "figures in `Practice/dataset_pairing_forensics_results/`. Do not promote the "
            "dataset unless the final decision table reports no unresolved pairing or "
            "orientation failures."
        )
    )

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
    )

    # Static safety checks only: do not execute the expensive forensic analysis.
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{NOTEBOOK_PATH.name}:cell-{index}")
            cell.execution_count = None
            cell.outputs = []

    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"created={NOTEBOOK_PATH}")
    print(f"cells={len(notebook.cells)} code_cells={sum(c.cell_type == 'code' for c in notebook.cells)}")


if __name__ == "__main__":
    main()
