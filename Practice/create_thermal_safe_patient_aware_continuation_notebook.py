"""Create a thermal-managed continuation notebook from the validated baseline."""

from copy import deepcopy
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Practice" / "baseline_25epoch_patient_aware_training.ipynb"
OUTPUT = ROOT / "Practice" / "continue_patient_aware_baseline_to_epoch10.ipynb"

notebook = nbformat.read(SOURCE, as_version=4)
notebook = deepcopy(notebook)

notebook.cells[0].source = """
# Thermal-Safe Patient-Aware Baseline Continuation

This notebook resumes the exact patient-aware baseline checkpoint produced by
`baseline_25epoch_patient_aware_training.ipynb`.

The prior run completed only epoch 1 and paused at 87°C. That is insufficient
for a model-revision decision. Continue in short sessions until at least epoch
10, then evaluate patient robustness, false positives, and lesion-size behavior.

The model, optimizer, scheduler, scaler, sampler generator, history, and all RNG
states are restored. The held-out test split remains locked.
""".strip()

notebook.cells[1].source = """
## tl;dr

1. Let the GPU cool below 84°C.
2. Run all cells.
3. Each invocation trains at most three additional epochs and stops sooner if
   the GPU reaches 88°C.
4. If fewer than 10 total epochs are complete, cool the GPU and rerun the
   training cell, then rerun the result cells below it.
5. Do not interpret `needs_model_revision` before epoch 10.

This notebook writes to the existing `patient_aware_baseline_outputs` directory
so the original experiment continues rather than starting a new model.
""".strip()

replacements = {
    "MAX_GPU_TEMP_C = 86": (
        "MAX_GPU_TEMP_C = 88\n"
        "MAX_START_GPU_TEMP_C = 84\n"
        "MAX_EPOCHS_PER_SESSION = 3"
    ),
    "THRESHOLDS = np.arange(0.30, 0.71, 0.05).round(2)": (
        "THRESHOLDS = np.arange(0.30, 0.91, 0.05).round(2)"
    ),
    "for epoch in range(start_epoch, EPOCHS + 1):": (
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
    ),
    'else:\n            stop_reason = "completed_25_epochs"': (
        "else:\n"
        "            stop_reason = (\n"
        "                'completed_25_epochs'\n"
        "                if session_end_epoch >= EPOCHS\n"
        "                else 'session_epoch_limit_reached'\n"
        "            )"
    ),
    'if history_frame.empty or best_patient_metrics.empty:\n    final_gate = {': (
        "minimum_decision_epoch_reached = (\n"
        "    not history_frame.empty\n"
        "    and int(history_frame['epoch'].max()) >= MIN_EPOCHS_BEFORE_EARLY_STOP\n"
        ")\n"
        "\n"
        "if not minimum_decision_epoch_reached:\n"
        "    completed_epochs = int(history_frame['epoch'].max()) if not history_frame.empty else 0\n"
        "    final_gate = {\n"
        "        'status': 'incomplete_continue_to_epoch_10',\n"
        "        'epochs_completed': completed_epochs,\n"
        "        'minimum_decision_epoch': MIN_EPOCHS_BEFORE_EARLY_STOP,\n"
        "        'test_images_accessed': False,\n"
        "        'decision': (\n"
        "            f'INCOMPLETE — {completed_epochs} epochs completed. Cool the GPU and '\n"
        "            f'continue to at least epoch {MIN_EPOCHS_BEFORE_EARLY_STOP}.'\n"
        "        ),\n"
        "    }\n"
        "elif history_frame.empty or best_patient_metrics.empty:\n"
        "    final_gate = {"
    ),
}

replacement_counts = {key: 0 for key in replacements}
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    for old, new in replacements.items():
        if old in cell.source:
            cell.source = cell.source.replace(old, new)
            replacement_counts[old] += 1
    cell.execution_count = None
    cell.outputs = []

missing = [key for key, count in replacement_counts.items() if count != 1]
if missing:
    raise RuntimeError(f"Expected exactly one replacement for: {missing}")

setup_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "code" and "PROJECT_ROOT = Path" in cell.source
)
notebook.cells[setup_index + 1:setup_index + 1] = [
    nbformat.v4.new_markdown_cell(
        """## Process Visualization

### Thermal-safe continuation workflow

This diagram shows what is restored, what runs during each short session, and
which evidence is required before a model decision."""
    ),
    nbformat.v4.new_code_cell(
        r"""
fig, axis = plt.subplots(figsize=(18, 4.8))
axis.set_xlim(0, 18)
axis.set_ylim(0, 5)
axis.axis("off")

steps = [
    (0.3, "Frozen manifest\n+ locked test", "#D9EAF7"),
    (3.2, "Load epoch-1\ncheckpoint", "#D9EAF7"),
    (6.1, "Restore optimizer,\nscheduler + RNG", "#FFF1CC"),
    (9.0, "Check GPU\nbelow 84 C", "#FFF1CC"),
    (11.9, "Train <=3 epochs\n+ full validation", "#DDEEDB"),
    (14.8, "Patient metrics,\nerrors + gate", "#F6D7D7"),
]
for x, label, color in steps:
    patch = plt.Rectangle(
        (x, 1.65), 2.35, 1.65, facecolor=color,
        edgecolor="#333333", linewidth=1.2
    )
    axis.add_patch(patch)
    axis.text(x + 1.175, 2.475, label, ha="center", va="center", fontsize=11)
for (left, _, _), (right, _, _) in zip(steps[:-1], steps[1:]):
    axis.annotate(
        "", xy=(right, 2.475), xytext=(left + 2.35, 2.475),
        arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": "#333333"},
    )
axis.text(
    9, 4.25,
    "Exact continuation -> epoch 10 minimum decision -> continue to 25 or revise",
    ha="center", va="center", fontsize=15, weight="bold",
)
axis.text(
    9, 0.65,
    "Primary selection: mean tumor-positive patient Dice | "
    "Guardrails: worst patient, empty-slice FP, positive-slice recall, temperature",
    ha="center", va="center", fontsize=10,
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "continuation_process_map.png", dpi=160, bbox_inches="tight")
plt.show()
"""
    ),
]

takeaway_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "markdown" and cell.source.startswith("## Takeaways")
)
notebook.cells[takeaway_index:takeaway_index] = [
    nbformat.v4.new_markdown_cell(
        """### Expanded process and error dashboard

The dashboard separates optimization progress, patient robustness, prediction
volume, false positives, confidence, and threshold behavior. It is regenerated
after every continuation session."""
    ),
    nbformat.v4.new_code_cell(
        r"""
if history_frame.empty:
    print("Expanded dashboard is available after at least one completed epoch.")
else:
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    # 1. Generalization gap.
    axes[0, 0].plot(
        history_frame["epoch"], history_frame["train_loss"],
        marker="o", label="Train loss", color="#2878B5"
    )
    axes[0, 0].plot(
        history_frame["epoch"], history_frame["val_loss"],
        marker="s", label="Validation loss", color="#F28E2B"
    )
    axes[0, 0].fill_between(
        history_frame["epoch"],
        history_frame["train_loss"],
        history_frame["val_loss"],
        color="#B8B8B8", alpha=0.25, label="Generalization gap",
    )
    axes[0, 0].set_title("Optimization and generalization gap")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # 2. Patient distribution at the current best epoch.
    if not best_patient_metrics.empty:
        positive_patients = best_patient_metrics.loc[
            best_patient_metrics["true_pixels"].gt(0)
        ].sort_values("micro_dice")
        colors = [
            "#E15759" if volume in {104, 116} else "#2878B5"
            for volume in positive_patients["volume_id"]
        ]
        axes[0, 1].barh(
            positive_patients["volume_id"].astype(str),
            positive_patients["micro_dice"], color=colors
        )
        axes[0, 1].axvline(
            positive_patients["micro_dice"].mean(),
            linestyle="--", color="#333333", label="Patient mean"
        )
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_title("Tumor-positive patient Dice")
        axes[0, 1].set_xlabel("Micro-Dice"); axes[0, 1].set_ylabel("Volume")
        axes[0, 1].legend()

        # 3. Predicted versus true burden by patient.
        max_pixels = max(
            positive_patients["true_pixels"].max(),
            positive_patients["predicted_pixels"].max(),
        )
        axes[0, 2].scatter(
            positive_patients["true_pixels"],
            positive_patients["predicted_pixels"],
            s=80, color="#76B7B2", edgecolor="#333333"
        )
        axes[0, 2].plot([1, max_pixels], [1, max_pixels], linestyle="--", color="#333333")
        for row in positive_patients.itertuples(index=False):
            axes[0, 2].annotate(
                str(row.volume_id), (row.true_pixels, row.predicted_pixels),
                xytext=(4, 4), textcoords="offset points", fontsize=8
            )
        axes[0, 2].set_xscale("log"); axes[0, 2].set_yscale("log")
        axes[0, 2].set_title("Predicted versus true tumor burden")
        axes[0, 2].set_xlabel("True pixels"); axes[0, 2].set_ylabel("Predicted pixels")

        # 4. False-positive rate for every patient.
        fp_sorted = best_patient_metrics.sort_values(
            "empty_slice_false_positive_pct", ascending=False
        )
        axes[1, 0].bar(
            fp_sorted["volume_id"].astype(str),
            fp_sorted["empty_slice_false_positive_pct"],
            color="#E15759",
        )
        axes[1, 0].axhline(15, linestyle="--", color="#333333", label="Guardrail 15%")
        axes[1, 0].set_title("Empty-slice false-positive rate by patient")
        axes[1, 0].set_xlabel("Volume"); axes[1, 0].set_ylabel("Empty slices with FP (%)")
        axes[1, 0].legend()

    # 5. Confidence versus Dice on tumor-positive slices.
    if not best_per_slice.empty:
        positive_slices = best_per_slice.loc[best_per_slice["true_pixels"].gt(0)]
        scatter = axes[1, 1].scatter(
            positive_slices["max_probability"], positive_slices["dice"],
            c=np.log10(positive_slices["true_pixels"].clip(lower=1)),
            cmap="viridis", alpha=0.45, s=22,
        )
        axes[1, 1].set_title("Confidence, Dice, and lesion size")
        axes[1, 1].set_xlabel("Maximum probability"); axes[1, 1].set_ylabel("Slice Dice")
        fig.colorbar(scatter, ax=axes[1, 1], label="log10(true tumor pixels)")

    # 6. Extended threshold trade-off.
    if not threshold_table.empty:
        axes[1, 2].plot(
            threshold_table["threshold"], threshold_table["micro_dice"],
            marker="o", label="Dice"
        )
        axes[1, 2].plot(
            threshold_table["threshold"], threshold_table["precision"],
            marker="s", label="Precision"
        )
        axes[1, 2].plot(
            threshold_table["threshold"], threshold_table["recall"],
            marker="^", label="Recall"
        )
        axes[1, 2].axvline(FIXED_THRESHOLD, linestyle="--", color="#333333")
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title("Threshold trade-off from 0.30 to 0.90")
        axes[1, 2].set_xlabel("Threshold"); axes[1, 2].set_ylabel("Metric")
        axes[1, 2].legend()

    fig.suptitle("Thermal-safe continuation analytical dashboard", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "continuation_expanded_dashboard.png", dpi=160, bbox_inches="tight")
    plt.show()
"""
    ),
]

notebook.metadata["continuation"] = {
    "source_notebook": SOURCE.name,
    "checkpoint_directory": "Practice/patient_aware_baseline_outputs",
    "minimum_decision_epoch": 10,
    "max_epochs_per_session": 3,
    "start_temperature_limit_c": 84,
    "post_epoch_temperature_limit_c": 88,
}

nbformat.validate(notebook)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
