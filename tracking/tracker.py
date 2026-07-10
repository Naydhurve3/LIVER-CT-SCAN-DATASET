"""Pipeline progress tracker — records runs, outputs, and errors to tracking/progress.json."""
import json, sys, time, subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

TRACKING_DIR = Path(__file__).parent
PROGRESS_FILE = TRACKING_DIR / "progress.json"

# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------
PIPELINE = [
    # (id, short_name, file, phase, depends_on)
    ("count",    "Count dataset files",     "tasks/organize_data.py --count",          "setup",   []),
    ("splits",   "Create stratified splits", "tasks/create_splits.py",                  "setup",   ["count"]),
    ("01",       "Data Exploration",        "studies/01_data_exploration.ipynb",        "eda",     ["splits"]),
    ("02",       "Decoder Debug",           "studies/02_decoder_debug.ipynb",           "debug",   ["04"]),
    ("03",       "Pilot Training",          "studies/03_pilot_training.ipynb",          "train",   ["01"]),
    ("04",       "Full Fine-Tune",          "studies/04_full_finetune.ipynb",           "train",   ["01"]),
    ("05",       "Train From Scratch",      "studies/05_train_scratch.ipynb",           "train",   ["01"]),
    ("06",       "Ensemble Eval",           "studies/06_ensemble_eval.ipynb",           "eval",    ["04", "05"]),
    ("07",       "Statistical Tests",       "studies/07_statistical_tests.ipynb",       "eval",    ["04", "05"]),
    ("08",       "FP Post-Processing",      "studies/08_fp_postprocessing.ipynb",       "eval",    ["04"]),
    ("09",       "Ablation Study",          "studies/09_ablation_study.ipynb",          "eval",    ["04"]),
    ("10",       "Paper Report",            "studies/10_paper_report.ipynb",            "report",  ["04", "05"]),
    ("tests",    "Test suite",              "python run.py tests",                      "qa",      []),
]

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _load() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"steps": {}, "pipeline": [p[0] for p in PIPELINE], "last_updated": None}

def _save(state: dict):
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    PROGRESS_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# Log a step result
# ---------------------------------------------------------------------------
def log(step_id: str, status: str, outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None, files_created: Optional[list] = None,
        note: Optional[str] = None):
    state = _load()
    entry = state["steps"].get(step_id, {"step_id": step_id, "file": "", "status": "pending"})
    # Find the file path from pipeline definition
    for p in PIPELINE:
        if p[0] == step_id:
            entry["file"] = p[2]
            entry["name"] = p[1]
            break
    entry["status"] = status
    if status == "running":
        entry["started"] = datetime.now(timezone.utc).isoformat()
    elif status in ("completed", "failed"):
        entry["finished"] = datetime.now(timezone.utc).isoformat()
    if outputs:
        entry.setdefault("outputs", {}).update(outputs)
    if error:
        entry.setdefault("errors", []).append(error)
    if files_created:
        entry.setdefault("files_created", []).extend(files_created)
    if note:
        entry["note"] = note
    state["steps"][step_id] = entry
    _save(state)
    _print_entry(step_id, entry)

# ---------------------------------------------------------------------------
# Run a step and auto-log result
# ---------------------------------------------------------------------------
def run_and_log(step_id: str, cmd: list):
    log(step_id, "running")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output_text = result.stdout + result.stderr
        if result.returncode == 0:
            log(step_id, "completed", outputs={"exit_code": 0})
            return True
        else:
            log(step_id, "failed", error=output_text[-2000:])
            return False
    except Exception as e:
        log(step_id, "failed", error=str(e))
        return False

# ---------------------------------------------------------------------------
# Show status
# ---------------------------------------------------------------------------
def show():
    state = _load()
    phases_order = ["setup", "eda", "train", "debug", "eval", "report", "qa"]
    phase_labels = {
        "setup": "SETUP",
        "eda": "EXPLORATION",
        "train": "TRAINING",
        "debug": "DEBUG",
        "eval": "EVALUATION",
        "report": "REPORT",
        "qa": "TESTS",
    }
    # Group by phase
    by_phase: Dict[str, list] = {}
    for p in PIPELINE:
        by_phase.setdefault(p[3], []).append(p)

    print("\n" + "=" * 70)
    print("  PIPELINE PROGRESS")
    print("=" * 70)
    total, done = 0, 0
    for ph in phases_order:
        if ph not in by_phase:
            continue
        entries = by_phase[ph]
        print(f"\n  [{phase_labels[ph]}]")
        for pid, name, fpath, _, _ in entries:
            total += 1
            step = state["steps"].get(pid, {})
            status = step.get("status", "pending")
            icon = {"completed": "OK", "failed": "FAIL", "running": "RUN", "skipped": "SKIP", "pending": "    "}
            disp = f"{icon.get(status, '    '):6s}"
            if status == "completed":
                done += 1
            outputs = step.get("outputs", {})
            out_str = ""
            if outputs:
                items = []
                for k, v in list(outputs.items())[:3]:
                    items.append(f"{k}={v}")
                out_str = "  |  " + "; ".join(items)
            print(f"    {pid:6s}  {disp}  {name:30s}{out_str}")
    print(f"\n  Progress: {done}/{total} steps completed")
    # Next step suggestion
    _suggest_next(state)
    print()

def _suggest_next(state: dict):
    """Find the next pending step whose dependencies are all completed."""
    completed = {pid for pid, s in state["steps"].items() if s.get("status") == "completed"}
    for pid, name, fpath, _, deps in PIPELINE:
        step = state["steps"].get(pid, {})
        if step.get("status") == "completed":
            continue
        if step.get("status") == "running":
            print(f"  >> Next: {name} (currently running)")
            return
        if all(d in completed for d in deps):
            print(f"  -> Suggested next: {name}  ({fpath})")
            return
    print("  ** All steps completed or blocked by incomplete dependencies")

def _print_entry(step_id: str, entry: dict):
    status = entry.get("status", "?")
    icon = {"completed": "OK", "failed": "FAIL", "running": "RUN", "skipped": "SKIP", "pending": "    "}
    name = entry.get("name", step_id)
    print(f"  {icon.get(status, '  ')} [{status.upper()}] {name}")

def _print_help():
    print("Usage: python tracking/tracker.py <command>")
    print()
    print("Commands:")
    print("  status               Show current pipeline progress")
    print("  next                 Suggest next step to run")
    print("  log <step> <status>  Manually log a step result")
    print("                       status: completed | failed | skipped | pending")
    print("  reset <step>         Reset a step to pending")
    print("  reset --all          Reset all steps")
    print("  help                 Show this message")
    print()
    print("Examples:")
    print("  python tracking/tracker.py status")
    print('  python tracking/tracker.py log 01 completed')
    print("  python tracking/tracker.py reset 02")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("help", "--help", "-h"):
        _print_help()
    elif args[0] == "status":
        show()
    elif args[0] == "next":
        state = _load()
        _suggest_next(state)
    elif args[0] == "log":
        if len(args) < 3:
            print("Usage: python tracking/tracker.py log <step_id> <status> [--outputs JSON]")
            sys.exit(1)
        extra = {}
        if "--outputs" in args:
            idx = args.index("--outputs")
            if idx + 1 < len(args):
                try:
                    extra = json.loads(args[idx + 1])
                except json.JSONDecodeError:
                    pass
        log(args[1], args[2], outputs=extra)
    elif args[0] == "reset":
        if len(args) > 1 and args[1] == "--all":
            for p in PIPELINE:
                log(p[0], "pending")
            print("All steps reset to pending")
        elif len(args) > 1:
            log(args[1], "pending")
            print(f"{args[1]} reset to pending")
        else:
            print("Usage: python tracking/tracker.py reset <step_id>")
    else:
        # Try running as a script — log-and-run
        run_and_log(args[0], args[1:])
