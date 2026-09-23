from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from laya_forge.calibration import fit_temperatures, softmax
from laya_forge.metrics import score, wilson
from laya_forge.model import Item
from laya_forge.policy import action_for, fit_thresholds
from laya_forge.spec import PolicyParams, Question, load_examples


def test_wilson_matches_reference_values() -> None:
    lo, hi = wilson(8, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.490, 0.943)
    assert wilson(0, 0) == (0.0, 1.0)
    assert wilson(10, 10)[1] == 1.0


def test_temperature_fit_recovers_the_true_temperature() -> None:
    rng = np.random.default_rng(0)
    true_t = 2.5
    logits, items = [], []
    for _ in range(3000):
        z = rng.normal(0, 3, 2)
        y = int(rng.random() < softmax(z, true_t)[1])
        logits.append(z)
        items.append(Item("q", 2, [], [0, 1], y))
    fitted = fit_temperatures(logits, items)
    assert fitted == {"noul:2": pytest.approx(true_t, abs=0.25)}


def test_thresholds_meet_their_budgets_on_the_fitting_data() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.2, 2000), 0, 1)
    params = PolicyParams(target_precision=0.95, max_miss_rate=0.05)
    th = fit_thresholds(p, y, params)
    blocked = p >= th["block_at"]
    assert y[blocked].mean() >= 0.95
    assert (p[y == 1] < th["allow_below"]).mean() <= 0.05
    assert th["allow_below"] <= th["block_at"]


def test_thresholds_never_block_without_positives() -> None:
    th = fit_thresholds(np.array([0.1, 0.9]), np.array([0, 0]), PolicyParams())
    assert th["block_at"] > 1 and action_for(0.99, th) != "block"


def test_score_reports_accuracy_and_calibration() -> None:
    items = [Item("q", 2, [], [0, 1], y) for y in (1, 1, 0, 0)]
    probs = [np.array([0.1, 0.9]), np.array([0.6, 0.4]), np.array([0.8, 0.2]), np.array([0.7, 0.3])]
    m = score(items, probs)["questions"]["q"]
    assert m["accuracy"] == 0.75 and m["n"] == 4
    assert 0 <= m["ece"] <= 1


@pytest.mark.parametrize(
    ("row", "error"),
    [
        ({"state": "x", "labels": {"alarm": "yes"}}, "expected true/false"),
        ({"state": "x", "labels": {"nope": True}}, "unknown question"),
        ({"state": "x", "labels": {"level": 7}}, "expected a level 0..2"),
        ({"state": "x", "labels": {"route": "moon"}}, "expected one of"),
        ({"state": "x"}, "labels"),
    ],
)
def test_bad_labels_fail_with_file_and_line(tmp_path: Path, row: dict, error: str) -> None:
    qs = {
        "alarm": Question(type="noul", instructions="?"),
        "level": Question(type="score", instructions="?", criteria=["a", "b", "c"]),
        "route": Question(type="choice", instructions="?", criteria={"fast": "x", "slow": "y"}),
    }
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps({"state": "ok", "labels": {"alarm": True}}) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=rf"(?s)d.jsonl:2: .*{error}"):
        load_examples(f, qs)


def test_choice_question_requires_criteria() -> None:
    with pytest.raises(ValueError, match="criteria"):
        Question(type="choice", instructions="?")


def test_train_layers_only_updates_the_top_of_the_encoder(tiny_checkpoint: Path) -> None:
    import torch

    from laya_forge.model import encode, load_agent, train
    from laya_forge.spec import Example, TrainParams

    agent = load_agent(str(tiny_checkpoint), "cpu")
    enc = agent.model.encoder
    before = {n: t.detach().clone() for n, t in enc.named_parameters()}
    q = {"alarm": Question(type="noul", instructions="fire?")}
    items = encode(agent, [Example(state="fire", labels={"alarm": True})] * 8, q)
    train(agent, items, TrainParams(epochs=1, train_layers=1, device="cpu"), log=lambda *_: None)
    changed = {n for n, t in enc.named_parameters() if not torch.equal(t, before[n])}
    assert changed and all(n.startswith(("layers.1.", "final_norm")) for n in changed)
