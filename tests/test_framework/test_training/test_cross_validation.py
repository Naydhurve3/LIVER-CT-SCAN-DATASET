from src.framework.training.cross_validation import kfold_split


def test_kfold_split_returns_correct_folds():
    volume_ids = list(range(20))
    folds = kfold_split(volume_ids, n_folds=5, seed=42)
    assert len(folds) == 5
    for fold in folds:
        assert "train" in fold
        assert "val" in fold
        assert len(fold["train"]) > 0
        assert len(fold["val"]) > 0


def test_kfold_split_disjoint():
    volume_ids = list(range(20))
    folds = kfold_split(volume_ids, n_folds=5, seed=42)
    for fold in folds:
        for v in fold["val"]:
            assert v not in fold["train"]


def test_kfold_split_all_volumes_covered():
    volume_ids = list(range(20))
    folds = kfold_split(volume_ids, n_folds=5, seed=42)
    all_val = []
    for fold in folds:
        all_val.extend(fold["val"])
    assert sorted(all_val) == volume_ids


def test_kfold_split_3_folds():
    volume_ids = list(range(9))
    folds = kfold_split(volume_ids, n_folds=3, seed=42)
    assert len(folds) == 3
    for fold in folds:
        assert len(fold["val"]) == 3
        assert len(fold["train"]) == 6
