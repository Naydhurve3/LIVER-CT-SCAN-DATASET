from argparse import Namespace
from pathlib import Path

from tools.train import _prepare_manifest, _resolve_directories


def _cfg(tmp_path):
    return {
        "experiment": {"name": "test-run"},
        "outputs": {
            "model_dir": str(tmp_path / "models"),
            "run_dir": str(tmp_path / "runs"),
        },
    }


def test_dry_run_isolated_by_timestamp(tmp_path):
    args = Namespace(output_dir=None, run_dir=None, dry_run=True)
    model_dir, run_dir = _resolve_directories(_cfg(tmp_path), args)
    assert model_dir.parent.name == "dry_run"
    assert run_dir.parent.name == "dry_run"


def test_existing_manifest_requires_resume_or_new_run_dir(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir = Path(cfg["outputs"]["run_dir"])
    args = Namespace(config="config.yaml", resume=None, dry_run=False)
    _prepare_manifest(run_dir, cfg, args, Path(cfg["outputs"]["model_dir"]))
    try:
        _prepare_manifest(run_dir, cfg, args, Path(cfg["outputs"]["model_dir"]))
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected existing manifest protection")


def test_resume_checkpoint_must_belong_to_model_directory(tmp_path):
    cfg = _cfg(tmp_path)
    foreign = tmp_path / "foreign" / "last_checkpoint.pth"
    foreign.parent.mkdir()
    foreign.touch()
    args = Namespace(config="config.yaml", resume=str(foreign), dry_run=False)
    try:
        _prepare_manifest(Path(cfg["outputs"]["run_dir"]), cfg, args,
                          Path(cfg["outputs"]["model_dir"]))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected cross-run checkpoint protection")
