from pathlib import Path
import nbformat

path = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\mark 1\mark_4d_metric_reconciliation_v116_diagnostic.ipynb")
notebook = nbformat.read(path, as_version=4)
replacements = {
    "prob=item['prob'].astype(np.float32)": "prob=item['probability'].astype(np.float32)",
    "cp=control_item['prob'][idx].astype(float)": "cp=control_item['probability'][idx].astype(float)",
    "rp=recall_item['prob'][idx].astype(float)": "rp=recall_item['probability'][idx].astype(float)",
}
counts = {old: 0 for old in replacements}
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    for old, new in replacements.items():
        if old in cell.source:
            counts[old] += cell.source.count(old)
            cell.source = cell.source.replace(old, new)
            cell.outputs = []
            cell.execution_count = None
if any(count != 1 for count in counts.values()):
    raise RuntimeError(f"Unexpected replacement counts: {counts}")
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Patched {path}")
