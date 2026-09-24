"""Phase 0 stand-in model for the feasibility spike.

The spike answers one question before any money is spent on training: can a
MiniLM-L6-sized encoder score sentences inside a Manifest V3 extension within
the latency and size budgets?  Size and speed depend on the architecture, not on
the weights, so this module builds a model with the exact MiniLM-L6 shape and
**random weights**.  Its predictions are meaningless and must never be reported
as quality numbers.

Output layout matches what transformers.js loads from a local folder::

    <out>/config.json
    <out>/tokenizer.json, tokenizer_config.json, ...
    <out>/onnx/model_quantized.onnx     # int8, dtype "q8" in transformers.js
    <out>/spike.json                    # provenance: seed, sizes, parity check
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

# MiniLM-L6-H384 (as in all-MiniLM-L6-v2): 6 layers, hidden 384, 12 heads.
MINILM_L6 = {
    "vocab_size": 30522,
    "hidden_size": 384,
    "num_hidden_layers": 6,
    "num_attention_heads": 12,
    "intermediate_size": 1536,
    "max_position_embeddings": 512,
}
LABELS = {0: "not_spoiler", 1: "spoiler"}
OPSET = 17


class SpikeReport(TypedDict):
    """Provenance and checks written to ``spike.json`` next to the model."""

    warning: str
    architecture: str
    seed: int
    parameters: int
    tokenizer_vocab_trained: int
    onnx_opset: int
    fp32_mb: float
    int8_mb: float
    max_abs_diff_torch_vs_onnx_fp32: float
    max_abs_diff_fp32_vs_int8: float
    max_abs_diff_torch_vs_onnx_other_shape: float


def _english_corpus() -> Iterator[str]:
    """Yield English text available offline: docstrings from the Python stdlib.

    Only used to train a throwaway WordPiece vocabulary so that tokenised
    sequence lengths are realistic.  The real student reuses its teacher's vocab.
    """
    stdlib = (
        Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    for path in sorted(stdlib.rglob("*.py")):
        if "test" in path.parts or "site-packages" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                doc = ast.get_docstring(node)
                if doc:
                    yield doc


def _train_tokenizer(out_dir: Path) -> int:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
    from tokenizers.trainers import WordPieceTrainer
    from transformers import BertTokenizer  # tokenizers-backed ("fast") in transformers 5

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    tok.train_from_iterator(
        _english_corpus(),
        WordPieceTrainer(vocab_size=MINILM_L6["vocab_size"], special_tokens=specials),
    )
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    fast = BertTokenizer(tokenizer_object=tok, model_max_length=128)
    fast.save_pretrained(out_dir)
    return tok.get_vocab_size()


def _file_mb(path: Path) -> float:
    return round(path.stat().st_size / 1_000_000, 2)


def build_spike_model(out_dir: str | Path, seed: int = 0) -> SpikeReport:
    """Build the random-weight MiniLM-L6 classifier and write it to ``out_dir``."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

    out = Path(out_dir)
    (out / "onnx").mkdir(parents=True, exist_ok=True)

    vocab_trained = _train_tokenizer(out)

    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=MINILM_L6["vocab_size"],
        hidden_size=MINILM_L6["hidden_size"],
        num_hidden_layers=MINILM_L6["num_hidden_layers"],
        num_attention_heads=MINILM_L6["num_attention_heads"],
        intermediate_size=MINILM_L6["intermediate_size"],
        max_position_embeddings=MINILM_L6["max_position_embeddings"],
        id2label=LABELS,  # two labels, so num_labels == 2
        label2id={v: k for k, v in LABELS.items()},
    )
    model = BertForSequenceClassification(config).eval()
    config.save_pretrained(out)
    n_params = sum(p.numel() for p in model.parameters())

    tokenizer = AutoTokenizer.from_pretrained(out)
    sample = tokenizer(
        ["Title: Severance", "Nobody expected her to be the one behind it all."],
        padding=True,
        return_tensors="pt",
    )
    inputs = (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"])

    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = Path(tmp) / "model.onnx"
        batch, seq = "batch", "sequence"
        torch.onnx.export(
            model,
            inputs,
            str(fp32_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: batch, 1: seq},
                "attention_mask": {0: batch, 1: seq},
                "token_type_ids": {0: batch, 1: seq},
                "logits": {0: batch},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        q8_path = out / "onnx" / "model_quantized.onnx"
        quantize_dynamic(str(fp32_path), str(q8_path), weight_type=QuantType.QInt8)

        feed = {k: v.numpy() for k, v in sample.items()}
        with torch.no_grad():
            torch_logits = model(**sample).logits.numpy()
        fp32_logits = np.asarray(ort.InferenceSession(str(fp32_path)).run(["logits"], feed)[0])
        q8_logits = np.asarray(ort.InferenceSession(str(q8_path)).run(["logits"], feed)[0])
        fp32_mb = _file_mb(fp32_path)

        # The exporter traces one input shape; prove other batch sizes and
        # lengths (with padding) still match PyTorch.
        other = tokenizer(
            [
                "Short one.",
                "A much longer sentence that forces padding for the others in this batch, "
                "so the attention mask actually matters for correctness here.",
                "Mid-length line about the finale.",
            ],
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            torch_other = model(**other).logits.numpy()
        onnx_other = np.asarray(
            ort.InferenceSession(str(fp32_path)).run(
                ["logits"], {k: v.numpy() for k, v in other.items()}
            )[0]
        )

    report: SpikeReport = {
        "warning": "Random weights. Predictions are meaningless; use for size and latency only.",
        "architecture": "BertForSequenceClassification, MiniLM-L6-H384 shape",
        "seed": seed,
        "parameters": n_params,
        "tokenizer_vocab_trained": vocab_trained,
        "onnx_opset": OPSET,
        "fp32_mb": fp32_mb,
        "int8_mb": _file_mb(q8_path),
        "max_abs_diff_torch_vs_onnx_fp32": float(np.abs(torch_logits - fp32_logits).max()),
        "max_abs_diff_fp32_vs_int8": float(np.abs(fp32_logits - q8_logits).max()),
        "max_abs_diff_torch_vs_onnx_other_shape": float(np.abs(torch_other - onnx_other).max()),
    }
    (out / "spike.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report
