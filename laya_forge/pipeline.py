"""`laya-forge run`: base scores, fine-tune, calibrate, fit thresholds, export, report, gate."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from . import __version__
from .calibration import fit_temperatures, probabilities
from .metrics import score
from .model import encode, load_agent, raw_logits, resolve_dir, save, served_temperatures, train
from .policy import evaluate_policy, fit_policy
from .report import render
from .spec import ForgeConfig, GateParams, load_examples

VARIANTS = {
    "base": "Laya as shipped",
    "base_calibrated": "Laya, calibration only (no training)",
    "forged_raw": "fine-tuned, uncalibrated",
    "forged": "fine-tuned + calibrated",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def check_gate(results: dict[str, Any], gate: GateParams, model: str = "forged") -> list[str]:
    """Every reason the model fails the gate, one line each. Empty means pass."""
    failures = []
    for name, res in results["test"].items():
        m, b = res[model]["overall"], res["base"]["overall"]
        if m["accuracy"] < gate.min_accuracy:
            failures.append(f"{name}: accuracy {m['accuracy']:.3f} < {gate.min_accuracy}")
        if m["ece"] > gate.max_ece:
            failures.append(f"{name}: ECE {m['ece']:.3f} > {gate.max_ece}")
        if gate.must_beat_base and m["accuracy"] < b["accuracy"]:
            failures.append(f"{name}: accuracy {m['accuracy']:.3f} below base {b['accuracy']:.3f}")
    return failures


def run(config_path: str | Path, log: Any = print) -> tuple[dict[str, Any], list[str]]:
    cfg = ForgeConfig.load(config_path)
    qs = cfg.questions
    train_ex = load_examples(cfg.train, qs)
    calib_ex = [ex for p in cfg.calibration for ex in load_examples(p, qs)]
    test_ex = {name: load_examples(p, qs) for name, p in cfg.test.items()}
    log(
        f"examples: train {len(train_ex)}, calibration {len(calib_ex)}, "
        + ", ".join(f"test/{k} {len(v)}" for k, v in test_ex.items())
    )

    agent = load_agent(cfg.base_model, cfg.train_params.device)
    log(f"loaded {cfg.base_model} on {agent.device}")
    train_items = encode(agent, train_ex, qs)
    calib_items = encode(agent, calib_ex, qs)
    test_items = {k: encode(agent, v, qs) for k, v in test_ex.items()}

    # Score the base model before training overwrites its weights in place.
    shipped = served_temperatures(agent)
    base_calib = raw_logits(agent, calib_items)
    base_temps = fit_temperatures(base_calib, calib_items)
    results: dict[str, Any] = {"test": {k: {} for k in test_items}}
    for k, items in test_items.items():
        z = raw_logits(agent, items)
        results["test"][k]["base"] = score(items, probabilities(z, items, shipped))
        results["test"][k]["base_calibrated"] = score(items, probabilities(z, items, base_temps))

    t0 = time.perf_counter()
    history = train(agent, train_items, cfg.train_params, log)
    train_s = time.perf_counter() - t0

    calib_z = raw_logits(agent, calib_items)
    temps = fit_temperatures(calib_z, calib_items)
    calib_probs = probabilities(calib_z, calib_items, temps)
    policy = fit_policy(calib_items, calib_probs, cfg.policy)
    log(f"fitted temperatures {temps}")
    for k, items in test_items.items():
        z = raw_logits(agent, items)
        results["test"][k]["forged_raw"] = score(items, probabilities(z, items, {}))
        probs = probabilities(z, items, temps)
        results["test"][k]["forged"] = score(items, probs)
        results["test"][k]["policy"] = evaluate_policy(items, probs, policy)

    results.update(
        {
            "laya_forge": __version__,
            "base_model": cfg.base_model,
            "device": str(agent.device),
            "train_params": cfg.train_params.model_dump(),
            "policy_params": cfg.policy.model_dump(),
            "train_loss": history,
            "train_seconds": round(train_s, 1),
            "temperatures": temps,
            "base_temperatures_fitted": base_temps,
            "policy": policy,
            "data": {
                "train": [len(train_items), _sha(cfg.train)],
                **{f"calibration/{p.stem}": [len(load_examples(p, qs)), _sha(p)] for p in cfg.calibration},
                **{f"test/{k}": [len(test_items[k]), _sha(p)] for k, p in cfg.test.items()},
            },
        }
    )
    failures = check_gate(results, cfg.gate)
    results["gate"] = {"params": cfg.gate.model_dump(), "failures": failures, "passed": not failures}

    out = cfg.output
    save(
        agent,
        resolve_dir(cfg.base_model),
        out / "model",
        temps,
        {
            "version": __version__,
            "base_model": cfg.base_model,
            "questions": {q: v.as_laya() for q, v in qs.items()},
            "policy": policy,
            "data": results["data"],
        },
    )
    (out / "report.json").write_text(json.dumps(results, indent=2, default=float))
    (out / "report.md").write_text(render(results))
    log(f"wrote {out / 'model'}, {out / 'report.md'}")
    return results, failures


def evaluate(model: str, config_path: str | Path, log: Any = print) -> tuple[dict[str, Any], list[str]]:
    """Score an existing checkpoint (as it would be served) against the base model, then gate it."""
    cfg = ForgeConfig.load(config_path)
    tests = {name: load_examples(p, cfg.questions) for name, p in cfg.test.items()}
    results: dict[str, Any] = {"test": {k: {} for k in tests}}
    for variant, path in (("base", cfg.base_model), ("forged", model)):
        agent = load_agent(path, cfg.train_params.device)
        temps = served_temperatures(agent)
        for k, ex in tests.items():
            items = encode(agent, ex, cfg.questions)
            results["test"][k][variant] = score(items, probabilities(raw_logits(agent, items), items, temps))
        log(f"scored {variant}: {path}")
    return results, check_gate(results, cfg.gate)
