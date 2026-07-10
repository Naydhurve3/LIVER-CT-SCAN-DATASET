from pathlib import Path
from typing import Any, Dict, Optional


class MetricsTracker:
    def __init__(self, output_dir: str = "experiments/logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._history: Dict[str, list] = {}

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        for key, value in metrics.items():
            if key not in self._history:
                self._history[key] = []
            entry = {"value": value}
            if step is not None:
                entry["step"] = step
            self._history[key].append(entry)

    def get_history(self, key: str) -> list:
        return self._history.get(key, [])

    def best(self, key: str, mode: str = "max"):
        values = [entry["value"] for entry in self._history.get(key, [])]
        if not values:
            return None
        return max(values) if mode == "max" else min(values)

    def save(self, path: Optional[str] = None):
        import json
        save_path = Path(path) if path else self.output_dir / "metrics.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(self._history, indent=2))

    def load(self, path: str):
        import json
        self._history = json.loads(Path(path).read_text())


class MLflowTracker:
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "medsegx",
                 run_name: Optional[str] = None):
        self._enabled = False
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri
        self._run = None

    def _ensure_mlflow(self):
        if not self._enabled:
            try:
                import mlflow
                self._mlflow = mlflow
                if self._tracking_uri:
                    mlflow.set_tracking_uri(self._tracking_uri)
                mlflow.set_experiment(self._experiment_name)
                self._enabled = True
            except ImportError:
                self._enabled = False

    def start_run(self, tags: Optional[Dict[str, str]] = None):
        self._ensure_mlflow()
        if self._enabled:
            self._run = self._mlflow.start_run(run_name=self._run_name)
            if tags:
                self._mlflow.set_tags(tags)

    def log_params(self, params: Dict[str, Any]):
        if self._enabled:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self._enabled:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str):
        if self._enabled:
            self._mlflow.log_artifact(local_path)

    def log_artifacts(self, local_dir: str):
        if self._enabled:
            self._mlflow.log_artifacts(local_dir)

    def end_run(self):
        if self._enabled and self._run:
            self._mlflow.end_run()
            self._run = None

    def log_calibration_plot(self, reliability_data: Dict[str, Any], output_path: str = "calibration.png"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.plot(reliability_data["bin_confidences"], reliability_data["bin_accuracies"],
                "o-", label="Model")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability Diagram (ECE={reliability_data['ece']:.4f})")
        ax.legend()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if self._enabled:
            self.log_artifact(output_path)
