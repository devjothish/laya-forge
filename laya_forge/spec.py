"""Typed boundary for everything a user hands the forge: the run config and the labelled examples.

Both are untrusted input. A label that does not fit its question (a string for a yes/no question,
a level past the end of a score scale) fails here with the file and line, not three frames deep
inside a training step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

State = str | dict[str, Any] | list[Any]
Label = bool | int | str


class Question(BaseModel):
    """One Laya question, in the same shape `laya.Agent.predict` takes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul", "choice", "score"]
    instructions: str
    criteria: dict[str, Any] | list[Any] | None = None

    @model_validator(mode="after")
    def _criteria_fit_type(self) -> Question:
        if self.type == "choice" and not self.criteria:
            raise ValueError("a choice question needs criteria: a dict of label -> description")
        if self.type == "score" and not isinstance(self.criteria, list):
            raise ValueError("a score question needs criteria: a list of level descriptions")
        return self

    def options(self) -> list[str]:
        """Answer labels in the order Laya scores them."""
        if self.type == "noul":
            return ["false", "true"]
        if self.type == "score":
            assert isinstance(self.criteria, list)
            return [str(i) for i in range(len(self.criteria))]
        assert self.criteria is not None
        return list(self.criteria) if isinstance(self.criteria, dict) else [str(c) for c in self.criteria]

    def label_index(self, label: Label) -> int:
        """Map a user label onto the option index the model is trained to pick."""
        if self.type == "noul":
            if not isinstance(label, bool):
                raise ValueError(f"expected true/false, got {label!r}")
            return int(label)
        if self.type == "score":
            n = len(self.options())
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < n:
                raise ValueError(f"expected a level 0..{n - 1}, got {label!r}")
            return label
        opts = self.options()
        if label not in opts:
            raise ValueError(f"expected one of {opts}, got {label!r}")
        return opts.index(str(label))

    def as_laya(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(2, ge=0)
    batch_size: int = Field(8, ge=1)
    lr: float = Field(2e-5, gt=0)
    head_lr: float = Field(1e-4, gt=0)
    # Encoder layers to fine-tune, counted from the top. None trains all of it; 0 trains only the
    # decision head. Fewer layers means less optimizer state: full fine-tuning of the 421M model
    # measured 12 GB on agentguard, top 12 layers 6.5 GB.
    train_layers: int | None = Field(None, ge=0)
    seed: int = 0
    device: str | None = None


class PolicyParams(BaseModel):
    """What the calibrated thresholds must guarantee on the calibration split."""

    model_config = ConfigDict(extra="forbid")

    target_precision: float = Field(0.95, gt=0, le=1)
    max_miss_rate: float = Field(0.05, ge=0, lt=1)


class GateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_accuracy: float = Field(0.0, ge=0, le=1)
    max_ece: float = Field(1.0, ge=0, le=1)
    must_beat_base: bool = True


class ForgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str = "convaiinnovations/laya"
    questions: dict[str, Question]
    train: Path
    # Labelled data that looks like production, for temperatures and thresholds. Not the
    # training distribution: on its own templates a fine-tuned model separates the classes
    # almost perfectly, and thresholds fitted there are far too aggressive on real traffic.
    calibration: list[Path]
    test: dict[str, Path]
    output: Path
    train_params: TrainParams = TrainParams()
    policy: PolicyParams = PolicyParams()
    gate: GateParams = GateParams()

    @classmethod
    def load(cls, path: str | Path) -> ForgeConfig:
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        if isinstance(raw.get("calibration"), str):
            raw["calibration"] = [raw["calibration"]]
        cfg = cls.model_validate(raw)
        # Data paths are relative to the config file, so a recipe runs from any working directory.
        base = path.parent
        cfg.train = base / cfg.train
        cfg.calibration = [base / p for p in cfg.calibration]
        cfg.test = {k: base / v for k, v in cfg.test.items()}
        cfg.output = base / cfg.output
        return cfg


class Example(BaseModel):
    """One labelled state. It may label any subset of the questions."""

    model_config = ConfigDict(extra="forbid")

    state: State
    labels: dict[str, Label]


def load_examples(path: str | Path, questions: dict[str, Question]) -> list[Example]:
    """Parse a JSONL file of examples, rejecting unknown questions and ill-typed labels by line."""
    out = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            ex = Example.model_validate(json.loads(line))
            for qid, label in ex.labels.items():
                if qid not in questions:
                    raise ValueError(f"unknown question {qid!r}; known: {sorted(questions)}")
                questions[qid].label_index(label)
        except ValueError as e:  # pydantic's ValidationError and json's JSONDecodeError are both
            raise ValueError(f"{path}:{n}: {e}") from e
        out.append(ex)
    if not out:
        raise ValueError(f"{path}: no examples")
    return out
