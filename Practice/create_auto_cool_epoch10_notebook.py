"""Create an auto-cooling continuation notebook through the epoch-10 decision."""

from copy import deepcopy
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Practice" / "continue_patient_aware_baseline_to_epoch10.ipynb"
OUTPUT = ROOT / "Practice" / "auto_cool_continue_patient_aware_to_epoch10.ipynb"

notebook = deepcopy(nbformat.read(SOURCE, as_version=4))

notebook.cells[0].source = """
# Auto-Cooling Patient-Aware Continuation to Epoch 10

This notebook resumes the existing patient-aware baseline from its latest
checkpoint and automatically waits for the GPU to cool between epochs.

The current evidence contains only two epochs. Continue to epoch 10 before
making a model, loss, sampling, or architecture decision.

The test split remains locked.
""".strip()

notebook.cells[1].source = """
## tl;dr

- Start with the GPU below 84°C.
- Run all cells once.
- Before each epoch, the notebook polls temperature every 30 seconds.
- Training resumes only below 84°C.
- At 88°C the notebook records a warning and cools before the next epoch.
- At 90°C or above it stops after saving the completed epoch.
- The run targets epoch 10, then produces the full patient-aware decision.

Cooldown observations are saved and visualized. The existing model, optimizer,
scheduler, mixed-precision scaler, sampler generator, RNG states, and histories
are restored exactly.
""".strip()

replacements = {
    (
        'axes[1, 2].axhline(MAX_GPU_TEMP_C, linestyle="--", color="#4D4D4D", label="Pause limit")'
    ): (
        'axes[1, 2].axhline(COOLDOWN_WARNING_TEMP_C, linestyle="--", color="#4D4D4D", label="Cooldown warning")'
    ),
    (
        "MAX_GPU_TEMP_C = 88\n"
        "MAX_START_GPU_TEMP_C = 84\n"
        "MAX_EPOCHS_PER_SESSION = 3"
    ): (
        "TARGET_DECISION_EPOCH = 10\n"
        "MAX_START_GPU_TEMP_C = 84\n"
        "COOLDOWN_WARNING_TEMP_C = 88\n"
        "EMERGENCY_STOP_TEMP_C = 90\n"
        "COOLDOWN_POLL_SECONDS = 30\n"
        "MAX_COOLDOWN_MINUTES = 30"
    ),
    (
        "session_end_epoch = min(EPOCHS, start_epoch + MAX_EPOCHS_PER_SESSION - 1)\n"
        "        for epoch in range(start_epoch, session_end_epoch + 1):\n"
        "            pre_epoch_thermal = gpu_stats()\n"
        "            if (\n"
        "                np.isfinite(pre_epoch_thermal['temperature_c'])\n"
        "                and pre_epoch_thermal['temperature_c'] >= MAX_START_GPU_TEMP_C\n"
        "            ):\n"
        "                stop_reason = 'waiting_for_cool_gpu'\n"
        "                print(\n"
        "                    f\"WAIT: GPU is {pre_epoch_thermal['temperature_c']:.0f}C before epoch \"\n"
        "                    f\"{epoch}. Cool below {MAX_START_GPU_TEMP_C}C and rerun this cell.\"\n"
        "                )\n"
        "                break"
    ): (
        "session_end_epoch = min(EPOCHS, TARGET_DECISION_EPOCH)\n"
        "        for epoch in range(start_epoch, session_end_epoch + 1):\n"
        "            cooled = wait_for_gpu_cool(epoch)\n"
        "            if not cooled:\n"
        "                stop_reason = 'cooldown_timeout'\n"
        "                print(\n"
        "                    f'WAIT: GPU did not cool below {MAX_START_GPU_TEMP_C}C within '\n"
        "                    f'{MAX_COOLDOWN_MINUTES} minutes. Check ventilation and rerun later.'\n"
        "                )\n"
        "                break"
    ),
    (
        'if np.isfinite(thermal["temperature_c"]) and thermal["temperature_c"] >= MAX_GPU_TEMP_C:\n'
        '                stop_reason = "paused_for_cooling"\n'
        '                print(f"PAUSED: GPU reached {thermal[\'temperature_c\']:.0f}C. Cool it, then rerun this cell.")\n'
        '                break'
    ): (
        'if np.isfinite(thermal["temperature_c"]) and thermal["temperature_c"] >= EMERGENCY_STOP_TEMP_C:\n'
        '                stop_reason = "emergency_thermal_stop"\n'
        '                print(\n'
        '                    f"STOP: GPU reached {thermal[\'temperature_c\']:.0f}C. "\n'
        '                    "The completed epoch is saved; inspect cooling before resuming."\n'
        '                )\n'
        '                break\n'
        '            if np.isfinite(thermal["temperature_c"]) and thermal["temperature_c"] >= COOLDOWN_WARNING_TEMP_C:\n'
        '                print(\n'
        '                    f"COOLDOWN: GPU reached {thermal[\'temperature_c\']:.0f}C. "\n'
        '                    "The next epoch will wait for a safe start temperature."\n'
        '                )'
    ),
    (
        "stop_reason = (\n"
        "                'completed_25_epochs'\n"
        "                if session_end_epoch >= EPOCHS\n"
        "                else 'session_epoch_limit_reached'\n"
        "            )"
    ): (
        "stop_reason = (\n"
        "                'completed_25_epochs'\n"
        "                if session_end_epoch >= EPOCHS\n"
        "                else 'epoch_10_decision_point_reached'\n"
        "            )"
    ),
}

counts = {key: 0 for key in replacements}
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    for old, new in replacements.items():
        if old in cell.source:
            cell.source = cell.source.replace(old, new)
            counts[old] += 1
    cell.execution_count = None
    cell.outputs = []

missing = [key for key, count in counts.items() if count != 1]
if missing:
    raise RuntimeError(f"Expected exactly one replacement for {len(missing)} block(s).")

utility_cell = next(
    cell for cell in notebook.cells
    if cell.cell_type == "code" and "def gpu_stats():" in cell.source
)
cooldown_code = r'''

def append_thermal_log(record: dict):
    path = OUTPUT_DIR / "thermal_cooldown_log.csv"
    frame = pd.DataFrame([record])
    if path.is_file():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def wait_for_gpu_cool(epoch: int) -> bool:
    """Wait in bounded 30-second checks until the next epoch is thermally safe."""
    if not torch.cuda.is_available():
        return True
    started_wait = time.time()
    max_wait_seconds = MAX_COOLDOWN_MINUTES * 60
    while True:
        thermal = gpu_stats()
        elapsed = time.time() - started_wait
        append_thermal_log({
            "timestamp": pd.Timestamp.now().isoformat(),
            "target_epoch": int(epoch),
            "temperature_c": thermal["temperature_c"],
            "utilization_pct": thermal["utilization_pct"],
            "memory_mb": thermal["memory_mb"],
            "cooldown_elapsed_seconds": elapsed,
        })
        temperature = thermal["temperature_c"]
        if not np.isfinite(temperature) or temperature < MAX_START_GPU_TEMP_C:
            print(
                f"READY: epoch {epoch} starts at "
                f"{temperature:.0f}C." if np.isfinite(temperature)
                else f"READY: epoch {epoch}; temperature unavailable."
            )
            return True
        if elapsed >= max_wait_seconds:
            return False
        print(
            f"COOLING: GPU {temperature:.0f}C; waiting "
            f"{COOLDOWN_POLL_SECONDS}s before epoch {epoch}."
        )
        time.sleep(COOLDOWN_POLL_SECONDS)
'''
insertion_point = utility_cell.source.index("\ndef evaluate_patient_aware")
utility_cell.source = (
    utility_cell.source[:insertion_point]
    + cooldown_code
    + utility_cell.source[insertion_point:]
)

takeaway_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "markdown" and cell.source.startswith("## Takeaways")
)
notebook.cells[takeaway_index:takeaway_index] = [
    nbformat.v4.new_markdown_cell(
        """### Thermal cooldown profile

This graph shows how long the GPU waited before each epoch and whether cooling
behavior is stable across the continuation run."""
    ),
    nbformat.v4.new_code_cell(
        r"""
thermal_log_path = OUTPUT_DIR / "thermal_cooldown_log.csv"
if thermal_log_path.is_file():
    thermal_log = pd.read_csv(thermal_log_path)
    thermal_log["sequence"] = np.arange(1, len(thermal_log) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for epoch, group in thermal_log.groupby("target_epoch"):
        axes[0].plot(
            group["sequence"], group["temperature_c"],
            marker="o", label=f"Before epoch {epoch}"
        )
    axes[0].axhline(
        MAX_START_GPU_TEMP_C, linestyle="--", color="#333333",
        label="Safe start boundary"
    )
    axes[0].set_title("GPU temperature during cooldown checks")
    axes[0].set_xlabel("Cooldown observation"); axes[0].set_ylabel("C")
    axes[0].legend(fontsize=8)

    final_wait = (
        thermal_log.groupby("target_epoch")["cooldown_elapsed_seconds"]
        .max().reset_index()
    )
    axes[1].bar(
        final_wait["target_epoch"].astype(str),
        final_wait["cooldown_elapsed_seconds"] / 60,
        color="#F28E2B",
    )
    axes[1].set_title("Cooldown time before each epoch")
    axes[1].set_xlabel("Target epoch"); axes[1].set_ylabel("Minutes")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "thermal_cooldown_profile.png", dpi=160, bbox_inches="tight")
    plt.show()
else:
    print("Thermal cooldown log will appear when continuation training starts.")
"""
    ),
]

notebook.metadata["auto_cooling"] = {
    "source_notebook": SOURCE.name,
    "target_decision_epoch": 10,
    "safe_start_temperature_c": 84,
    "warning_temperature_c": 88,
    "emergency_stop_temperature_c": 90,
    "poll_seconds": 30,
    "max_cooldown_minutes": 30,
}

nbformat.validate(notebook)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
