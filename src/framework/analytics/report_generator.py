from pathlib import Path


class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def generate_dataset_report(self, stats, output_name="dataset_report"):
        lines = []
        lines.append("=" * 60)
        lines.append("DATASET REPORT")
        lines.append("=" * 60)
        lines.append("")
        for key, value in stats.items():
            lines.append(f"{key}: {value}")
        path = self.output_dir / f"{output_name}.txt"
        self._write(path, "\n".join(lines))
        return str(path)

    def generate_model_report(self, metrics, output_name="model_report"):
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        for key, value in metrics.items():
            lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        path = self.output_dir / f"{output_name}.txt"
        self._write(path, "\n".join(lines))
        return str(path)

    def generate_html_report(self, title, sections, output_name="report"):
        html = [f"<html><head><title>{title}</title></head><body>"]
        html.append(f"<h1>{title}</h1>")
        for section_name, content in sections.items():
            html.append(f"<h2>{section_name}</h2>")
            html.append(f"<pre>{content}</pre>")
        html.append("</body></html>")
        path = self.output_dir / f"{output_name}.html"
        self._write(path, "\n".join(html))
        return str(path)

    def generate_json_report(self, data, output_name="report"):
        import json
        path = self.output_dir / f"{output_name}.json"
        self._write(path, json.dumps(data, indent=2))
        return str(path)
