"""Spike model export: size budget and ONNX parity. Needs the 'train' extra."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from spoiler_shield.export.spike import build_spike_model

pytestmark = pytest.mark.train


def test_spike_model_meets_budget_and_matches_pytorch(tmp_path):
    report = build_spike_model(tmp_path, seed=0)

    assert (tmp_path / "onnx" / "model_quantized.onnx").exists()
    assert (tmp_path / "tokenizer.json").exists()
    assert json.loads((tmp_path / "config.json").read_text())["num_hidden_layers"] == 6

    assert report["int8_mb"] <= 30, "model must fit the 30 MB download budget"
    assert report["max_abs_diff_torch_vs_onnx_fp32"] < 1e-4
    assert report["max_abs_diff_torch_vs_onnx_other_shape"] < 1e-4
    assert report["max_abs_diff_fp32_vs_int8"] < 0.1
