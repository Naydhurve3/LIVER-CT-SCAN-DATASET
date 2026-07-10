import json
from pathlib import Path
from src.framework.training.tracker import MetricsTracker


def test_metrics_tracker_log_and_get():
    tracker = MetricsTracker()
    tracker.log({"loss": 0.5, "dice": 0.85})
    tracker.log({"loss": 0.3, "dice": 0.88})
    history = tracker.get_history("dice")
    assert len(history) == 2
    assert history[0]["value"] == 0.85
    assert history[1]["value"] == 0.88


def test_metrics_tracker_best():
    tracker = MetricsTracker()
    tracker.log({"dice": 0.85})
    tracker.log({"dice": 0.92})
    tracker.log({"dice": 0.88})
    assert tracker.best("dice", mode="max") == 0.92
    assert tracker.best("loss", mode="min") is None


def test_metrics_tracker_save_load(tmp_path):
    tracker = MetricsTracker()
    tracker.log({"dice": 0.85})
    path = tmp_path / "metrics.json"
    tracker.save(str(path))
    assert path.exists()
    data = json.loads(path.read_text())
    assert "dice" in data

    tracker2 = MetricsTracker()
    tracker2.load(str(path))
    assert len(tracker2.get_history("dice")) == 1


def test_metrics_tracker_empty_history():
    tracker = MetricsTracker()
    assert tracker.get_history("nonexistent") == []
    assert tracker.best("nonexistent") is None
