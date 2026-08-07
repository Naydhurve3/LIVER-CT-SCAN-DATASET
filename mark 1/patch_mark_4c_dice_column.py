from pathlib import Path
import nbformat

path = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\mark 1\mark_4c_two_channel_recall_ablation.ipynb")
notebook = nbformat.read(path, as_version=4)
old_variants = [
    "control_patients=pd.read_csv(MARK4_DIR/'best_validation_patient_metrics.csv'); control_patients['arm']='control'; patient_frames.append(control_patients[['volume_id','dice','arm']])",
    "control_patients=pd.read_csv(MARK4_DIR/'best_validation_patient_metrics.csv').rename(columns={'micro_dice':'dice'}); control_patients['arm']='control'; patient_frames.append(control_patients[['volume_id','dice','arm']])",
]
new = """control_patients=pd.read_csv(MARK4_DIR/'best_validation_patient_metrics.csv').rename(columns={'micro_dice':'dice'})
assert {'volume_id','dice'}.issubset(control_patients.columns), control_patients.columns.tolist()
control_patients['arm']='control'; patient_frames.append(control_patients[['volume_id','dice','arm']])"""
matches = [(cell, old) for cell in notebook.cells if cell.cell_type == "code" for old in old_variants if old in cell.source]
if len(matches) != 1:
    raise RuntimeError(f"Expected exactly one target cell, found {len(matches)}")
cell, old = matches[0]
cell.source = cell.source.replace(old, new)
cell.outputs = []
cell.execution_count = None
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Patched {path}")
