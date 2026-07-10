"""S20: Generate paper-ready tables and summaries from evaluation results."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.utils import setup_logging, logger


def load_results():
    base = Path(__file__).resolve().parent.parent
    paths = {
        "ensemble": base / "results" / "ensemble_evaluation.json",
        "stats": base / "data" / "metadata" / "statistics.json",
    }
    data = {}
    for key, p in paths.items():
        if p.exists():
            data[key] = json.loads(p.read_text())
    return data


def make_latex_table(results):
    """Produce a LaTeX table comparing models across metrics."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Quantitative comparison of segmentation performance.}",
        "\\label{tab:results}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & Dice $\\uparrow$ & IoU $\\uparrow$ & HD95 $\\downarrow$ & ASD $\\downarrow$ & NSD $\\uparrow$ \\\\",
        "\\midrule",
    ]
    model_order = ["best_model", "member_0", "member_1", "member_2", "ensemble"]
    model_names = {
        "best_model": "MobileNetV2-UNet",
        "member_0": "UP\\textsuperscript{3}RE-M0",
        "member_1": "UP\\textsuperscript{3}RE-M1",
        "member_2": "UP\\textsuperscript{3}RE-M2",
        "ensemble": "UP\\textsuperscript{3}RE-Ensemble",
    }
    if "ensemble" in results:
        for key in model_order:
            if key not in results:
                continue
            r = results[key]
            row = f"  {model_names.get(key, key)}"
            for m in ["dice", "iou", "hd95", "asd", "nsd"]:
                v = r.get(m, {})
                if v.get("mean") is not None:
                    row += f" & ${v['mean']:.3f}\\pm{v['std']:.3f}$"
                else:
                    row += " & ---"
            row += " \\\\"
            lines.append(row)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def make_clinical_summary(stats):
    if not stats:
        return ""
    ts = stats.get("tumor_statistics", {})
    intensity = stats.get("intensity_statistics", {})
    ds = stats.get("dataset_summary", {})
    lines = [
        "\\section*{Dataset Summary}",
        f"Total volumes: {ds.get('total_volumes', 'N/A')}",
        f"Total slices: {ds.get('total_slices', 'N/A')}",
        f"Tumor-positive slices: {ts.get('slices_with_tumor', 'N/A')} ({ts.get('tumor_slice_pct', 'N/A')}\\%)",
        f"Tumor pixel proportion: {ts.get('tumor_pixel_pct', 'N/A')}\\%",
        f"Class imbalance ratio: {ts.get('imbalance_ratio', 'N/A')}:1",
        f"Mean intensity: {intensity.get('mean', 'N/A'):.4f} $\\pm$ {intensity.get('std', 'N/A'):.4f}",
    ]
    has_tumor = sum(1 for v in ts.get("per_volume", {}).values() if v.get("slices_with_tumor", 0) > 0)
    lines.append(f"Volumes with tumor: {has_tumor}/{ds.get('total_volumes', 'N/A')}")
    return "\n".join(lines)


def make_md_report(results, stats):
    lines = [
        "# Liver Tumor Segmentation — Evaluation Report",
        "",
        "## 1. Dataset",
        "",
    ]
    if stats:
        ts = stats.get("tumor_statistics", {})
        ds = stats.get("dataset_summary", {})
        lines.extend([
            f"- **Volumes:** {ds.get('total_volumes', 'N/A')}",
            f"- **Slices:** {ds.get('total_slices', 'N/A')} (256$\\times$256)",
            f"- **Tumor slices:** {ts.get('slices_with_tumor', 'N/A')}/{ts.get('total_slices', 'N/A')} ({ts.get('tumor_slice_pct', 'N/A')}\\%)",
            f"- **Imbalance:** {ts.get('imbalance_ratio', 'N/A')}:1",
            "",
        ])

    lines.extend([
        "",
        "## 2. Quantitative Results",
        "",
        "| Model | Dice | IoU | HD95 | ASD | NSD |",
        "|-------|------|-----|------|-----|-----|",
    ])
    model_order = ["best_model", "member_0", "member_1", "member_2", "ensemble"]
    model_names = {
        "best_model": "MobileNetV2-UNet",
        "member_0": "UP³RE-M0",
        "member_1": "UP³RE-M1",
        "member_2": "UP³RE-M2",
        "ensemble": "UP³RE-Ensemble",
    }
    if "ensemble" in results:
        for key in model_order:
            if key not in results:
                continue
            r = results[key]
            name = model_names.get(key, key)
            row = f"| {name} "
            for m in ["dice", "iou", "hd95", "asd", "nsd"]:
                v = r.get(m, {})
                if v.get("mean") is not None:
                    row += f"| {v['mean']:.4f}$\\pm${v['std']:.4f} "
                else:
                    row += "| --- "
            row += "|"
            lines.append(row)
    lines.extend([
        "",
        "## 3. Key Findings",
        "",
        "- TBD after evaluation",
        "",
    ])
    return "\n".join(lines)


def generate():
    data = load_results()
    info = []

    if "ensemble" in data:
        table = make_latex_table(data["ensemble"])
        report_dir = Path(__file__).resolve().parent.parent / "results"
        (report_dir / "latex_table.tex").write_text(table)
        info.append("LaTeX table: results/latex_table.tex")

        md = make_md_report(data["ensemble"], data.get("stats"))
        (report_dir / "evaluation_report.md").write_text(md)
        info.append("Markdown report: results/evaluation_report.md")

    if "stats" in data:
        clin = make_clinical_summary(data["stats"])
        report_dir = Path(__file__).resolve().parent.parent / "results"
        (report_dir / "clinical_summary.tex").write_text(clin)
        info.append("Clinical summary: results/clinical_summary.tex")

    for line in info:
        print(line)
    logger.info("Paper report generated")
    return info


if __name__ == "__main__":
    setup_logging()
    generate()
