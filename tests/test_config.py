import pytest
from pathlib import Path
from src.utils.config_loader import load_config

def test_configs_loads():
    cfg = load_config("config/config.yaml")
    assert cfg is not None

def test_config_has_required_keys():
    cfg = load_config("config/config.yaml")
    assert "device" in cfg
    assert "category" in cfg
    assert "model" in cfg
    assert "threshold" in cfg

def test_config_device_valid():
    cfg = load_config("config/config.yaml")
    assert cfg["device"] in ["cuda", "cpu"]

def test_config_threshold_percentile_valid():
    cfg = load_config("config/config.yaml")
    percentile =  cfg["threshold"]["percentile"]
    assert 70 <= percentile <= 100

def test_config_category_is_string():
    cfg = load_config("config/config.yaml")
    assert isinstance(cfg["category"], str)

def test_config_model_backbone():
    cfg = load_config("config/config.yaml")
    assert "backbone" in cfg["model"]
    assert cfg["model"]["backbone"] in ["resnet50", "wide_resnet50_2"]