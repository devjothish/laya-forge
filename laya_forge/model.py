"""Everything that touches Laya's weights: encode, score, fine-tune, save.

Sequences are built with Laya's own `Agent._encode_state`, so a checkpoint trained here sees
exactly the token layout it will be served with. Those are private Laya internals, which is why
`laya` is pinned to a minor version in pyproject.toml and the round trip is covered by a test.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from laya.agent import Agent
from laya.common import QTYPES, collate_items, temp_bucket
from safetensors import safe_open
from safetensors.torch import save_file

from .spec import Example, Question, TrainParams

_CHECKPOINT_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]


@dataclass
class Item:
    """One (state, question) pair, tokenized, with the index of the right answer."""

    qid: str
    qtype: int
    ids: list[int]
    markers: list[int]
    label: int

    @property
    def k(self) -> int:
        return len(self.markers)

    @property
    def bucket(self) -> str:
        return str(temp_bucket(self.qtype, self.k))


def resolve_dir(model: str) -> Path:
    """Local directory holding a Laya checkpoint, downloading only its own files if needed."""
    if os.path.isdir(model):
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(str(snapshot_download(model, allow_patterns=_CHECKPOINT_FILES)))


def load_agent(model: str, device: str | None = None) -> Agent:
    with warnings.catch_warnings():
        # The stock checkpoint ships one out-of-range temperature and says so on every load.
        # The forge reports calibration itself, so the warning is noise here.
        warnings.filterwarnings("ignore", message="laya: this checkpoint ships invalid")
        return Agent(model, device=device)


def encode(agent: Agent, examples: list[Example], questions: dict[str, Question]) -> list[Item]:
    internal = {qid: agent._to_internal(q.as_laya()) for qid, q in questions.items()}
    items = []
    for ex in examples:
        for qid, label in ex.labels.items():
            (raw,) = agent._encode_state(ex.state, [qid], internal)
            items.append(
                Item(qid, raw["qtype"], raw["ids"], raw["markers"], questions[qid].label_index(label))
            )
    return items


def _batch(agent: Agent, items: list[Item]) -> dict[str, Any]:
    raw = [{"ids": it.ids, "markers": it.markers, "qtype": it.qtype, "label": it.label} for it in items]
    b = collate_items([raw], agent.tok.pad_token_id)
    return {k: v.to(agent.device) for k, v in b.items() if isinstance(v, torch.Tensor)}


def _forward(agent: Agent, b: dict[str, Any]) -> torch.Tensor:
    logits, _ = agent.model(
        b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"]
    )
    return logits  # type: ignore[no-any-return]


@torch.no_grad()
def raw_logits(agent: Agent, items: list[Item], batch_size: int = 32) -> list[np.ndarray]:
    """Untempered logits per item, one entry per answer option."""
    agent.model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        logits = _forward(agent, _batch(agent, chunk)).float().cpu().numpy()
        out.extend(logits[j, : it.k] for j, it in enumerate(chunk))
    return out


def served_temperatures(agent: Agent) -> dict[str, float]:
    """The temperature Laya applies to each bucket at inference, after its own clamping."""
    temps = {
        f"{name}:{size}": agent.temperature[qt]
        for name, qt in QTYPES.items()
        for size in ("2", "3-5", "6-10", "11+")
    }
    temps.update(agent.temperature_by_options)
    return temps


def train(agent: Agent, items: list[Item], p: TrainParams, log: Any = print) -> list[float]:
    """Supervised fine-tune on the answer index. Returns the mean loss of each epoch."""
    torch.manual_seed(p.seed)
    rng = random.Random(p.seed)
    model = agent.model
    enc = model.encoder
    if p.train_layers is None:
        enc.requires_grad_(True)
    else:
        enc.requires_grad_(False)
        if p.train_layers:
            for layer in enc.layers[-p.train_layers :]:
                layer.requires_grad_(True)
            enc.final_norm.requires_grad_(True)
    head = [q for n, q in model.named_parameters() if not n.startswith("encoder.")]
    groups = [{"params": head, "lr": p.head_lr}]
    tuned = [q for q in enc.parameters() if q.requires_grad]
    if tuned:
        groups.append({"params": tuned, "lr": p.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = max(1, p.epochs * math.ceil(len(items) / p.batch_size))
    warmup = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / (steps - warmup + 1)))
    )

    history = []
    for epoch in range(p.epochs):
        model.train()
        order = items[:]
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), p.batch_size):
            b = _batch(agent, order[i : i + p.batch_size])
            loss = torch.nn.functional.cross_entropy(_forward(agent, b), b["label"])
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            if len(losses) % 50 == 0:
                log(f"  epoch {epoch + 1} step {len(losses)}: loss {np.mean(losses[-50:]):.4f}")
        history.append(float(np.mean(losses)))
        log(f"epoch {epoch + 1}/{p.epochs}: mean loss {history[-1]:.4f}")
    model.eval()
    return history


def save(
    agent: Agent, base_dir: Path, out: Path, temperatures: dict[str, float], meta: dict[str, Any]
) -> None:
    """Write a checkpoint that plain `laya.load(out)` and `laya-serve` accept."""
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("tokenizer", "encoder"):
        if (base_dir / sub).exists():
            shutil.copytree(base_dir / sub, out / sub, dirs_exist_ok=True)

    with safe_open(str(base_dir / "model.safetensors"), "pt") as f:
        dtype = f.get_tensor(next(iter(f.keys()))).dtype
    state = {
        k: (v.to(dtype) if v.is_floating_point() else v).detach().cpu().contiguous()
        for k, v in agent.model.state_dict().items()
    }
    save_file(state, str(out / "model.safetensors"))

    cfg = dict(agent.cfg)
    # Only fitted buckets are overwritten. Stock buckets outside Laya's accepted range are
    # replaced by the value Laya already clamps them to, so the checkpoint loads without a warning.
    stock = served_temperatures(agent)
    cfg["temperature_by_options"] = {k: stock[k] for k in cfg.get("temperature_by_options", {})}
    cfg["temperature_by_options"].update(temperatures)
    cfg["temperature"] = [agent.temperature[i] for i in range(3)]
    cfg["laya_forge"] = meta
    (out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
