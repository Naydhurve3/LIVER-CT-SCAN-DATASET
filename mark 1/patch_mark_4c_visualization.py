from pathlib import Path
import nbformat

path = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\mark 1\mark_4c_two_channel_recall_ablation.ipynb")
notebook = nbformat.read(path, as_version=4)
old = "control_patients=pd.read_csv(MARK4_DIR/'best_validation_patient_metrics.csv'); control_patients['arm']='control'"
new = "control_patients=pd.read_csv(MARK4_DIR/'best_validation_patient_metrics.csv').rename(columns={'micro_dice':'dice'}); control_patients['arm']='control'"
matches = 0
for cell in notebook.cells:
    if cell.cell_type == "code" and old in cell.source:
        cell.source = cell.source.replace(old, new)
        matches += 1
if matches != 1:
    raise RuntimeError(f"Expected one visualization cell match, found {matches}")
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Patched {path}")
