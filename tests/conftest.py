"""A tiny, randomly initialised checkpoint in Laya's exact on-disk format.

It uses the real Laya tokenizer (a few MB) and Laya's own `build_model`, so the whole pipeline -
encode, train, calibrate, save, `laya.load`, Guard - runs through the same code paths as the
421M model, in seconds and on CPU.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from laya.common import build_model
from safetensors.torch import save_file

TOKENIZER_REPO = "convaiinnovations/laya"


def _record(state: str, label: bool) -> str:
    return json.dumps({"state": {"text": state}, "labels": {"alarm": label}})


def write_split(path: Path, n: int, offset: int = 0) -> Path:
    """A trivially learnable yes/no task: does the text contain the word 'fire'?"""
    words = ["river", "stone", "cloud", "paper", "green", "table", "music", "light"]
    rows = []
    for i in range(offset, offset + n):
        w = [words[(i * 3 + j) % len(words)] for j in range(3)]
        label = i % 2 == 0
        if label:
            w.insert(i % 3, "fire")
        rows.append(_record(" ".join(w), label))
    path.write_text("\n".join(rows) + "\n")
    return path


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from transformers import AutoTokenizer, ModernBertConfig

    out = tmp_path_factory.mktemp("tiny_laya")
    src = Path(snapshot_download(TOKENIZER_REPO, allow_patterns=["tokenizer/*"]))
    shutil.copytree(src / "tokenizer", out / "tokenizer")
    tok = AutoTokenizer.from_pretrained(out / "tokenizer")

    enc = ModernBertConfig(
        vocab_size=len(tok),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=256,
        pad_token_id=tok.pad_token_id,
        cls_token_id=tok.cls_token_id,
        sep_token_id=tok.sep_token_id,
        bos_token_id=tok.cls_token_id,
        eos_token_id=tok.sep_token_id,
    )
    (out / "encoder").mkdir()
    enc.save_pretrained(out / "encoder")
    cfg = {
        "encoder": "answerdotai/ModernBERT-large",
        "head_layers": 1,
        "max_len": 128,
        "head_max_len": 64,
        "act_costs": {"escalate": 0.5},
        "amp_dtype": "fp16",
        "temperature": [1.0, 1.0, 1.0],
        # Out of Laya's accepted range, like the stock checkpoint's choice:11+ bucket.
        "temperature_by_options": {"choice:11+": 0.1},
    }
    (out / "rl_agent_config.json").write_text(json.dumps(cfg))
    model = build_model(cfg, encoder_dir=str(out / "encoder"), pretrained=False)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    return out


@pytest.fixture(scope="module")
def recipe(tmp_path_factory: pytest.TempPathFactory, tiny_checkpoint: Path) -> Path:
    """A complete forge config over the toy task, pointing at the tiny checkpoint."""
    tmp_path = tmp_path_factory.mktemp("recipe")
    write_split(tmp_path / "train.jsonl", 64)
    write_split(tmp_path / "calib.jsonl", 32, offset=100)
    write_split(tmp_path / "test.jsonl", 32, offset=200)
    cfg = {
        "base_model": str(tiny_checkpoint),
        "questions": {"alarm": {"type": "noul", "instructions": "Does `text` mention fire?"}},
        "train": "train.jsonl",
        "calibration": "calib.jsonl",
        "test": {"toy": "test.jsonl"},
        "output": "out",
        "train_params": {"epochs": 10, "batch_size": 8, "lr": 1e-3, "head_lr": 1e-3, "device": "cpu"},
        "gate": {"min_accuracy": 0.0, "max_ece": 1.0, "must_beat_base": False},
    }
    import yaml

    (tmp_path / "forge.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path / "forge.yaml"
