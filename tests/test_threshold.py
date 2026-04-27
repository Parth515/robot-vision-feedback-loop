import pytest 
import torch
from unittest.mock import MagicMock
from src.anomaly.threshold import compute_threshold
from PIL import Image

# Build a lightweight mock PatchCore for threshold tests.
def make_mock_model(score_values):
    model = MagicMock()
    model.transform = MagicMock(side_effect=lambda img: torch.randn(3, 224, 224))
    model.score = MagicMock(side_effect=score_values)
    return model

def test_threshold_is_float(tmp_path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    for i in range(5):
        Image.new("RGB", (64,64), color=(200, 200, 200)).save(good_dir/f"{i:03d}.png")
    scores = [0.1, 0.2, 0.15, 0.3, 0.25]
    mock_model = make_mock_model(scores)
    threshold = compute_threshold(mock_model, str(good_dir), percentile=95)
    assert isinstance(threshold, float)

def test_threshold_respects_percentile(tmp_path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    for i in range(10):
        Image.new("RGB", (64, 64)).save(good_dir / f"{i:03d}.png")

    scores_50 = list(range(1, 11))  # known values
    scores_95 = list(range(1, 11))

    mock_50 = make_mock_model(scores_50)
    mock_95 = make_mock_model(scores_95)

    t50 = compute_threshold(mock_50, str(good_dir), percentile=50)
    t95 = compute_threshold(mock_95, str(good_dir), percentile=95)

    assert t50 < t95

def test_threshold_positive():
    with pytest.raises(Exception):
        compute_threshold(None, "nonexistent_dir", percentile=95)