from pathlib import Path

import nbformat


ROOT = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")
SOURCE = ROOT / "Practice" / "multitask_liver_tumor_localization.ipynb"
DESTINATION = ROOT / "Practice" / "multitask_liver_tumor_epoch10_continuation.ipynb"

notebook = nbformat.read(SOURCE, as_version=4)

notebook.cells[0].source = """# Multi-task liver and tumor continuation to epoch 10

## tl;dr

This notebook resumes the existing multi-task checkpoint from the next incomplete epoch. It does **not** change the architecture, loss, optimizer, preprocessing, sampler, or validation population before the epoch-10 decision point.

The continuation adds bounded thermal waiting and detailed per-epoch progress visualization. The test split remains locked."""

notebook.cells[1].source = """## Context & Methods

### Key assumptions

- `Practice/multitask_liver_tumor_outputs/multitask_last.pth` is the authoritative continuation checkpoint.
- Epoch 1 alone is insufficient to judge the multi-task design.
- Resume must restore the model, optimizer, scheduler, scaler, sampler, and random-number state.
- Ground-truth liver masks remain training supervision only.
- Validation preprocessing and prediction gating do not use ground-truth organ masks.
- The test split remains inaccessible.

### Decision point

Continue through epoch 10 while tracking patient Dice, liver Dice, volumes 104/116, missed-positive slices, empty-slice false positives, loss, learning rate, elapsed time, and GPU temperature."""

setup_cell = notebook.cells[2]
setup_cell.source = setup_cell.source.replace(
    "EMERGENCY_STOP_TEMP_C = 90",
    """EMERGENCY_STOP_TEMP_C = 90
COOLDOWN_POLL_SECONDS = 30
MAX_COOLDOWN_MINUTES = 30""",
)

helper_cell = notebook.cells[12]
helper_cell.source += r'''


def gpu_stats():
    if not torch.cuda.is_available():
        return {"temperature_c": np.nan, "utilization_pct": np.nan, "memory_mb": np.nan}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True,
        )
        values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
        return {
            "temperature_c": values[0],
            "utilization_pct": values[1],
            "memory_mb": values[2],
        }
    except Exception:
        return {"temperature_c": np.nan, "utilization_pct": np.nan, "memory_mb": np.nan}


def wait_for_gpu_cool(epoch):
    """Wait in bounded intervals instead of ending the continuation."""
    if not torch.cuda.is_available():
        return True
    started = time.time()
    thermal_rows = []
    while True:
        stats = gpu_stats()
        elapsed = time.time() - started
        thermal_rows.append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "target_epoch": int(epoch),
            "temperature_c": stats["temperature_c"],
            "utilization_pct": stats["utilization_pct"],
            "memory_mb": stats["memory_mb"],
            "cooldown_elapsed_seconds": elapsed,
        })
        thermal_path = OUTPUT_DIR / "multitask_thermal_log.csv"
        new_rows = pd.DataFrame(thermal_rows[-1:])
        if thermal_path.is_file():
            new_rows = pd.concat([pd.read_csv(thermal_path), new_rows], ignore_index=True)
        new_rows.to_csv(thermal_path, index=False)
        temperature = stats["temperature_c"]
        if not np.isfinite(temperature) or temperature < MAX_START_GPU_TEMP_C:
            print(
                f"READY: epoch {epoch} starts at {temperature:.0f}C."
                if np.isfinite(temperature)
                else f"READY: epoch {epoch}; GPU temperature unavailable."
            )
            return True
        if elapsed >= MAX_COOLDOWN_MINUTES * 60:
            print(
                f"PAUSE: GPU remained at {temperature:.0f}C or above for "
                f"{MAX_COOLDOWN_MINUTES} minutes. The checkpoint is unchanged."
            )
            return False
        print(
            f"COOLING: GPU {temperature:.0f}C; waiting "
            f"{COOLDOWN_POLL_SECONDS} seconds before epoch {epoch}."
        )
        time.sleep(COOLDOWN_POLL_SECONDS)


def save_epoch_progress(history):
    """Save a bounded six-panel progress dashboard after every completed epoch."""
    frame = pd.DataFrame(history).copy()
    if frame.empty:
        return
    figure, axes = plt.subplots(2, 3, figsize=(20, 11))

    axes[0, 0].plot(frame["epoch"], frame["train_total_loss"], marker="o", label="Train")
    axes[0, 0].plot(frame["epoch"], frame["val_loss"], marker="s", label="Validation")
    axes[0, 0].set_title("Total loss by epoch")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss"); axes[0, 0].legend()

    axes[0, 1].plot(
        frame["epoch"], frame["gated_mean_patient_dice"],
        marker="o", color="#2878B5", label="Mean patient Dice",
    )
    axes[0, 1].axhline(TARGETS["mean_patient_dice"], linestyle="--",
                       color="#4D4D4D", label="Required")
    axes[0, 1].set_ylim(0, 1); axes[0, 1].set_title("Patient Dice progress")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend()

    axes[0, 2].plot(
        frame["epoch"], frame["mean_liver_dice"],
        marker="o", color="#4E9F3D", label="Liver Dice",
    )
    axes[0, 2].axhline(TARGETS["mean_liver_dice"], linestyle="--",
                       color="#4D4D4D", label="Required")
    axes[0, 2].set_ylim(0, 1); axes[0, 2].set_title("Predicted-liver progress")
    axes[0, 2].set_xlabel("Epoch"); axes[0, 2].legend()

    axes[1, 0].plot(
        frame["epoch"], frame["gated_positive_predicted_empty_pct"],
        marker="o", label="Positive predicted empty",
    )
    axes[1, 0].plot(
        frame["epoch"], frame["gated_empty_slice_false_positive_pct"],
        marker="s", label="Empty-slice FP",
    )
    axes[1, 0].axhline(TARGETS["positive_predicted_empty_pct"], linestyle="--",
                       color="#777777", label="Positive-empty limit")
    axes[1, 0].axhline(TARGETS["empty_slice_false_positive_pct"], linestyle=":",
                       color="#222222", label="Empty-FP limit")
    axes[1, 0].set_ylim(0, 100); axes[1, 0].set_title("Slice-level error rates")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Slices (%)")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        frame["epoch"], frame["gated_volume_104_dice"],
        marker="o", label="Volume 104",
    )
    axes[1, 1].plot(
        frame["epoch"], frame["gated_volume_116_dice"],
        marker="s", label="Volume 116",
    )
    axes[1, 1].axhline(TARGETS["volume_104_dice"], linestyle="--",
                       color="#2878B5", alpha=0.7, label="V104 required")
    axes[1, 1].axhline(TARGETS["volume_116_dice"], linestyle=":",
                       color="#F28E2B", label="V116 required")
    axes[1, 1].set_ylim(0, 1); axes[1, 1].set_title("Focus-patient recovery")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Micro-Dice")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(frame["epoch"], frame["lr"], marker="o",
                    color="#6F4E7C", label="Learning rate")
    axes[1, 2].set_yscale("log"); axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Learning rate"); axes[1, 2].set_title("Schedule and GPU temperature")
    if "temperature_c" in frame:
        temperature_axis = axes[1, 2].twinx()
        temperature_axis.plot(frame["epoch"], frame["temperature_c"],
                              marker="s", color="#F28E2B", label="GPU temperature")
        temperature_axis.axhline(MAX_START_GPU_TEMP_C, linestyle="--", color="#777777")
        temperature_axis.set_ylabel("GPU temperature (C)")

    figure.suptitle(
        f"Multi-task continuation progress — completed epoch {int(frame['epoch'].max())}",
        fontsize=17, weight="bold",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "multitask_epoch_progress.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    latest = frame.iloc[-1]
    status = {
        "last_completed_epoch": int(latest["epoch"]),
        "decision_epoch": DECISION_EPOCH,
        "epochs_remaining": max(DECISION_EPOCH - int(latest["epoch"]), 0),
        "mean_patient_dice": float(latest["gated_mean_patient_dice"]),
        "mean_liver_dice": float(latest["mean_liver_dice"]),
        "volume_104_dice": float(latest["gated_volume_104_dice"]),
        "volume_116_dice": float(latest["gated_volume_116_dice"]),
        "positive_predicted_empty_pct": float(
            latest["gated_positive_predicted_empty_pct"]
        ),
        "empty_slice_false_positive_pct": float(
            latest["gated_empty_slice_false_positive_pct"]
        ),
        "temperature_c": (
            float(latest["temperature_c"])
            if "temperature_c" in latest and np.isfinite(latest["temperature_c"])
            else None
        ),
        "test_images_accessed": False,
    }
    (OUTPUT_DIR / "multitask_continuation_status.json").write_text(
        json.dumps(status, indent=2)
    )
'''

training_cell = notebook.cells[16]
old_temperature_block = '''    if torch.cuda.is_available():
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        temperature = float(query.stdout.splitlines()[0]) if query.returncode == 0 else np.nan
        if np.isfinite(temperature) and temperature >= MAX_START_GPU_TEMP_C:
            print(f"GPU is {temperature:.0f}C; stop now and rerun after cooling.")
            break'''
new_temperature_block = '''    cooled = wait_for_gpu_cool(epoch)
    if not cooled:
        break
    epoch_started = time.perf_counter()'''
if old_temperature_block not in training_cell.source:
    raise RuntimeError("Could not locate the original temperature block.")
training_cell.source = training_cell.source.replace(
    old_temperature_block, new_temperature_block
)
training_cell.source = training_cell.source.replace(
    '''    record = {
        "epoch": epoch,''',
    '''    thermal = gpu_stats()
    record = {
        "epoch": epoch,
        "temperature_c": thermal["temperature_c"],
        "gpu_utilization_pct": thermal["utilization_pct"],
        "gpu_memory_mb": thermal["memory_mb"],
        "epoch_elapsed_seconds": time.perf_counter() - epoch_started,''',
)
training_cell.source = training_cell.source.replace(
    '''    pd.DataFrame(history).to_csv(OUTPUT_DIR/"multitask_history.csv", index=False)
    print(f"epoch={epoch:02d} patient={score:.4f} liver={validation['mean_liver_dice']:.4f} "''',
    '''    pd.DataFrame(history).to_csv(OUTPUT_DIR/"multitask_history.csv", index=False)
    save_epoch_progress(history)
    print(f"epoch={epoch:02d} patient={score:.4f} liver={validation['mean_liver_dice']:.4f} "''',
)

progress_markdown = nbformat.v4.new_markdown_cell(
    """### 8. Continuation status and live progress

The graph below is regenerated after every completed epoch. Target lines show the epoch-10 decision requirements. The status table separates current values from remaining epochs."""
)
progress_code = nbformat.v4.new_code_cell(
    r'''history_frame = pd.read_csv(OUTPUT_DIR / "multitask_history.csv")
save_epoch_progress(history_frame.to_dict("records"))
progress_path = OUTPUT_DIR / "multitask_epoch_progress.png"
if progress_path.is_file():
    display(Image.open(progress_path))

latest = history_frame.iloc[-1]
tracking = pd.DataFrame([
    {"measure": "Completed epoch", "actual": int(latest["epoch"]), "target": DECISION_EPOCH},
    {"measure": "Mean patient Dice", "actual": latest["gated_mean_patient_dice"],
     "target": TARGETS["mean_patient_dice"]},
    {"measure": "Mean liver Dice", "actual": latest["mean_liver_dice"],
     "target": TARGETS["mean_liver_dice"]},
    {"measure": "Volume 104 Dice", "actual": latest["gated_volume_104_dice"],
     "target": TARGETS["volume_104_dice"]},
    {"measure": "Volume 116 Dice", "actual": latest["gated_volume_116_dice"],
     "target": TARGETS["volume_116_dice"]},
    {"measure": "Positive predicted empty (%)",
     "actual": latest["gated_positive_predicted_empty_pct"],
     "target": TARGETS["positive_predicted_empty_pct"]},
    {"measure": "Empty-slice FP (%)",
     "actual": latest["gated_empty_slice_false_positive_pct"],
     "target": TARGETS["empty_slice_false_positive_pct"]},
])
display(tracking.style.format({"actual": "{:.4f}", "target": "{:.4f}"}))
if int(latest["epoch"]) < DECISION_EPOCH:
    print(f"CONTINUE: {DECISION_EPOCH-int(latest['epoch'])} epochs remain before the decision.")
else:
    print("DECISION POINT REACHED: interpret the full validation gate below.")'''
)

notebook.cells.insert(17, progress_markdown)
notebook.cells.insert(18, progress_code)

nbformat.write(notebook, DESTINATION)
print(f"Wrote {DESTINATION}")
