"""Liver Tumor Segmentation — Unified Pipeline

USAGE:
  python run.py --tasks       Run quick task scripts
  python run.py --tools       List CLI tools available
  python run.py --studies     Open studies/ with Jupyter
  python run.py --status      Show pipeline progress
  python run.py tests         Run all tests
"""
import sys, os, subprocess, json
from pathlib import Path

ROOT = Path(__file__).parent
_DS_GPU = str(ROOT.parent.parent / "ENVIRONMENTS" / "ds_gpu" / "Scripts")
if _DS_GPU not in os.environ["PATH"]:
    os.environ["PATH"] = _DS_GPU + os.pathsep + os.environ["PATH"]
PYTHON = sys.executable

TRACKER = ROOT / "tracking" / "tracker.py"

TASKS = {
    "count":   ("Count dataset files",     ["tasks/organize_data.py", "--count"]),
    "export":  ("Export file list JSON",   ["tasks/organize_data.py", "--export"]),
    "splits":  ("Create stratified splits",["tasks/create_splits.py"]),
}

TOOLS = {
    "train":    "tools/train.py",
    "validate": "tools/validate.py",
    "analyze":  "tools/analyze.py",
    "evaluate": "tools/evaluate.py",
    "ablate":   "tools/ablate.py",
    "report":   "tools/report.py",
}

STUDIES = [
    ("01", "Data Exploration & Canonical Stats",  "10 min"),
    ("02", "Decoder Artifact Debugging",           "5 min"),
    ("03", "Pilot Training (5 epochs)",            "5 min"),
    ("04", "Full Fine-Tune (50 epochs)",           "4 hrs"),
    ("05", "Train From Scratch (50 epochs)",       "5 hrs"),
    ("06", "Ensemble Evaluation",                  "10 min"),
    ("07", "Statistical Tests",                    "5 min"),
    ("08", "FP Reduction & Post-Processing",       "15 min"),
    ("09", "Ablation Study",                       "30 min"),
    ("10", "Paper Report Generation",              "2 min"),
]

TASK_STEP_IDS = {"count": "count", "export": "export", "splits": "splits"}

def _log_step(step_id, status, extra=None):
    cmd = [PYTHON, str(TRACKER), "log", step_id, status]
    if extra:
        cmd += ["--outputs", json.dumps(extra)]
    subprocess.run(cmd, capture_output=True)

def main():
    print("=" * 60)
    print("LIVER TUMOR SEGMENTATION — Unified Pipeline")
    print("=" * 60)
    print(f"Python: {PYTHON}")
    print(f"Root:   {ROOT}")
    print()

    if "--status" in sys.argv:
        subprocess.run([PYTHON, str(TRACKER), "status"])

    elif "--tasks" in sys.argv:
        for name, (desc, cmd) in TASKS.items():
            step_id = TASK_STEP_IDS.get(name, name)
            print(f"\n--- {name}: {desc} ---")
            _log_step(step_id, "running")
            r = subprocess.run([PYTHON] + cmd, capture_output=True, text=True)
            if r.returncode == 0:
                _log_step(step_id, "completed")
            else:
                _log_step(step_id, "failed", extra={"error": r.stderr[-500:]})
            print(r.stdout)
            if r.stderr:
                print(r.stderr)

    elif "--tools" in sys.argv:
        print("CLI TOOLS:")
        for name, path in TOOLS.items():
            print(f"  python {path}  ({name})")
        print("\nExample: python tools/train.py --help")

    elif "--studies" in sys.argv:
        print("STUDIES NOTEBOOKS:")
        for num, title, duration in STUDIES:
            print(f"  {num}. {title:40s} {duration:>8s}")
        print("\nLaunch: jupyter notebook studies/")

    elif "--log" in sys.argv and len(sys.argv) >= 4:
        step_id = sys.argv[2]
        status = sys.argv[3]
        extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
        _log_step(step_id, status, extra)

    elif "tests" in sys.argv:
        _log_step("tests", "running")
        r = subprocess.run([PYTHON, "-m", "pytest", str(ROOT / "tests"), "-v"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            _log_step("tests", "completed")
        else:
            _log_step("tests", "failed", extra={"error": r.stderr[-500:]})
        print(r.stdout)
        if r.stderr:
            print(r.stderr)

    else:
        print("QUICK TASKS:    python run.py --tasks")
        print("CLI TOOLS:      python run.py --tools")
        print("STUDIES:        python run.py --studies")
        print("TESTS:          python run.py tests")
        print("STATUS:         python run.py --status")
        print()
        print("See tracking/ for detailed pipeline status")

if __name__ == "__main__":
    main()
