"""The forge's promise, end to end: what it exports is a normal Laya checkpoint that serves exactly
the calibrated probabilities the forge measured, and the Guard applies the fitted thresholds."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import laya
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from laya_forge import Guard
from laya_forge.calibration import probabilities
from laya_forge.cli import main
from laya_forge.model import encode, load_agent, raw_logits, served_temperatures
from laya_forge.pipeline import run
from laya_forge.spec import ForgeConfig, load_examples

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture(scope="module")
def forged(recipe: Path) -> tuple[Path, dict]:
    results, _ = run(recipe, log=lambda *_: None)
    return recipe.parent / "out" / "model", results


def test_training_reduces_loss(forged: tuple[Path, dict]) -> None:
    _, results = forged
    assert results["train_loss"][-1] < results["train_loss"][0]


def test_export_loads_in_plain_laya_without_warnings(forged: tuple[Path, dict]) -> None:
    model_dir, results = forged
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the out-of-range stock temperature must be repaired
        agent = laya.load(str(model_dir), device="cpu")
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
    assert cfg["temperature_by_options"]["choice:11+"] == 0.5
    for bucket, t in results["temperatures"].items():
        assert agent.temperature_by_options[bucket] == pytest.approx(t)


def test_laya_serves_the_probabilities_the_forge_measured(forged: tuple[Path, dict], recipe: Path) -> None:
    model_dir, _ = forged
    cfg = ForgeConfig.load(recipe)
    examples = load_examples(cfg.test["toy"], cfg.questions)
    agent = load_agent(str(model_dir), "cpu")
    items = encode(agent, examples, cfg.questions)
    ours = probabilities(raw_logits(agent, items), items, served_temperatures(agent))
    q = {"alarm": cfg.questions["alarm"].as_laya()}
    for ex, p in zip(examples, ours, strict=True):
        served = agent.predict(ex.state, q)["answers"]["alarm"]["noul"]
        assert served == pytest.approx(p[1], abs=1e-3)


def test_guard_applies_thresholds_and_traces(forged: tuple[Path, dict]) -> None:
    model_dir, results = forged
    guard = Guard(model_dir, device="cpu")
    _exporter.clear()
    v = guard.check({"text": "the fire spread"})
    th = results["policy"]["alarm"]
    p = v.probabilities["alarm"]
    expected = "block" if p >= th["block_at"] else "allow" if p < th["allow_below"] else "escalate"
    assert v.action == expected
    assert v.allowed == (expected == "allow")
    (span,) = _exporter.get_finished_spans()
    assert span.name == "laya_forge.guard.check"
    assert span.attributes["laya_forge.action"] == expected
    assert span.attributes["laya_forge.p.alarm"] == pytest.approx(p)


def test_shadow_mode_never_blocks(forged: tuple[Path, dict]) -> None:
    model_dir, _ = forged
    guard = Guard(model_dir, device="cpu", shadow=True)
    guard.policy["alarm"] = {"allow_below": 0.0, "block_at": 0.0}  # everything would block
    v = guard.check({"text": "anything"})
    assert v.action == "block" and v.allowed


def test_guard_failure_is_a_verdict_not_an_exception(forged: tuple[Path, dict]) -> None:
    model_dir, _ = forged
    guard = Guard(model_dir, device="cpu", on_error="block")
    guard.questions["alarm"] = {"type": "noul"}  # no instructions: Laya rejects the question
    v = guard.check({"text": "x"})
    assert v.action == "block" and v.error


def test_guard_rejects_checkpoint_without_policy(tiny_checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="no laya-forge policy"):
        Guard(tiny_checkpoint, device="cpu")


def test_cli_exit_code_follows_gate(recipe: Path) -> None:
    import yaml

    assert main(["run", str(recipe)]) == 0
    cfg = yaml.safe_load(recipe.read_text())
    cfg["gate"] = {"max_ece": 0.0, "min_accuracy": 1.0}
    strict = recipe.parent / "strict.yaml"  # data paths resolve relative to the config
    strict.write_text(yaml.safe_dump(cfg))
    assert main(["eval", str(recipe.parent / "out" / "model"), str(strict)]) == 1
    assert (recipe.parent / "out" / "report.md").read_text().startswith("# laya-forge report")


def test_choice_questions_train_calibrate_and_serve(tiny_checkpoint: Path, tmp_path: Path) -> None:
    import yaml

    colors = ["red", "green", "blue"]
    rows = [
        json.dumps({"state": {"text": f"the {c} one, item {i}"}, "labels": {"color": c}})
        for i in range(30)
        for c in colors
    ]
    (tmp_path / "d.jsonl").write_text("\n".join(rows) + "\n")
    cfg = {
        "base_model": str(tiny_checkpoint),
        "questions": {
            "color": {"type": "choice", "instructions": "Which colour?", "criteria": {c: c for c in colors}}
        },
        "train": "d.jsonl",
        "calibration": ["d.jsonl"],
        "test": {"toy": "d.jsonl"},
        "output": "out",
        "train_params": {"epochs": 1, "device": "cpu"},
    }
    (tmp_path / "f.yaml").write_text(yaml.safe_dump(cfg))
    results, _ = run(tmp_path / "f.yaml", log=lambda *_: None)
    assert set(results["temperatures"]) == {"choice:3-5"}
    assert results["policy"] == {}  # thresholds are for yes/no questions only
    agent = laya.load(str(tmp_path / "out" / "model"), device="cpu")
    assert agent.temperature_by_options["choice:3-5"] == pytest.approx(results["temperatures"]["choice:3-5"])
    answer = agent.predict({"text": "the red one"}, {"color": cfg["questions"]["color"]})["answers"]["color"]
    assert answer["choice"] in colors
