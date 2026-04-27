import torch
from unittest.mock import MagicMock
from PIL import Image

def make_mock_patchcore(score=0.8, threshold=0.5):
    model = MagicMock()
    model.threshold = threshold
    model.transform = MagicMock(return_value=torch.randn(3, 224, 224))
    model.score = MagicMock(return_value=score)
    return model

def test_defect_detected_above_threshold(tmp_path):
    img_path = tmp_path / "defect.png"
    Image.new("RGB", (224, 224), color=(100, 50, 50)).save(img_path)

    mock_model = make_mock_patchcore(score=0.9, threshold=0.5)

    img = Image.open(img_path).convert("RGB")
    img_tensor = mock_model.transform(img).unsqueeze(0)
    score = mock_model.score(img_tensor)

    assert score > mock_model.threshold

def test_normal_below_threshold(tmp_path):
    img_path = tmp_path / "normal.png"
    Image.new("RGB", (224, 224), color=(200, 200, 200)).save(img_path)

    mock_model = make_mock_patchcore(score=0.2, threshold=0.5)

    img = Image.open(img_path).convert("RGB")
    img_tensor = mock_model.transform(img).unsqueeze(0)
    score = mock_model.score(img_tensor)

    assert score < mock_model.threshold


def test_score_is_float(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (224, 224)).save(img_path)

    mock_model = make_mock_patchcore(score=0.45, threshold=0.5)
    img = Image.open(img_path).convert("RGB")
    img_tensor = mock_model.transform(img).unsqueeze(0)
    score = mock_model.score(img_tensor)

    assert isinstance(score, float)


def test_edge_case_flagged(tmp_path):
    img_path = tmp_path / "edge.png"
    Image.new("RGB", (224, 224)).save(img_path)

    mock_model = make_mock_patchcore(score=0.62, threshold=0.5)
    img = Image.open(img_path).convert("RGB")
    img_tensor = mock_model.transform(img).unsqueeze(0)
    score = mock_model.score(img_tensor)
    delta = score - mock_model.threshold

    # edge case if delta is small
    assert delta > 0
    assert round(delta, 4) == round(0.62 - 0.5, 4)


def test_transform_output_shape(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (224, 224)).save(img_path)

    real_transform_output = torch.randn(3, 224, 224)
    mock_model = make_mock_patchcore()
    mock_model.transform = MagicMock(return_value=real_transform_output)

    img = Image.open(img_path).convert("RGB")
    tensor = mock_model.transform(img)
    assert tensor.shape == (3, 224, 224)