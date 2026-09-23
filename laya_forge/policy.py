"""Turn calibrated yes/no probabilities into allow / escalate / block, and apply that in production.

A single 0.5 cutoff forces every case into allow or block. Two thresholds, each fitted to a
stated error budget on held-out labels, leave a middle band that goes to a slower check
(an LLM, a human) instead of being guessed:

    p < allow_below               allow     misses at most `max_miss_rate` of true positives
    allow_below <= p < block_at   escalate
    p >= block_at                 block     at least `target_precision` of blocks are right
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from opentelemetry import trace

from .model import Item, load_agent
from .spec import PolicyParams, State

Action = Literal["allow", "escalate", "block"]
_SEVERITY: dict[Action, int] = {"allow": 0, "escalate": 1, "block": 2}
NEVER = 1.01  # a threshold no probability reaches

tracer = trace.get_tracer("laya_forge")


def fit_thresholds(p_true: np.ndarray, y: np.ndarray, params: PolicyParams) -> dict[str, float]:
    """The two thresholds for one yes/no question, from calibrated P(true) and 0/1 labels."""
    cands = np.unique(np.concatenate([p_true, [NEVER]]))
    block_at = NEVER
    for t in cands:  # ascending: the first threshold that meets the precision target
        sel = p_true >= t
        if sel.any() and y[sel].mean() >= params.target_precision:
            block_at = float(t)
            break
    pos = p_true[y == 1]
    allow_below = 0.0
    if len(pos):
        for t in cands[::-1]:  # descending: the highest threshold within the miss budget
            if (pos < t).mean() <= params.max_miss_rate:
                allow_below = float(t)
                break
    return {"allow_below": min(allow_below, block_at), "block_at": block_at}


def action_for(p: float, th: dict[str, float]) -> Action:
    if p >= th["block_at"]:
        return "block"
    return "allow" if p < th["allow_below"] else "escalate"


def _noul_groups(
    items: Sequence[Item], probs: Sequence[np.ndarray]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        if it.k == 2 and it.qtype == 2:
            groups[it.qid].append(i)
    return {
        q: (np.array([probs[i][1] for i in idx]), np.array([items[i].label for i in idx]))
        for q, idx in groups.items()
    }


def fit_policy(items: Sequence[Item], probs: Sequence[np.ndarray], params: PolicyParams) -> dict[str, Any]:
    return {q: fit_thresholds(p, y, params) for q, (p, y) in _noul_groups(items, probs).items()}


def evaluate_policy(
    items: Sequence[Item], probs: Sequence[np.ndarray], policy: dict[str, Any]
) -> dict[str, Any]:
    """What the thresholds do on a labelled set: where cases land and which ones land wrong."""
    out = {}
    for q, (p, y) in _noul_groups(items, probs).items():
        acts = [action_for(float(pi), policy[q]) for pi in p]
        n = len(acts)
        out[q] = {
            "n": n,
            "allow": acts.count("allow") / n,
            "escalate": acts.count("escalate") / n,
            "block": acts.count("block") / n,
            "missed_positives": sum(a == "allow" and yi == 1 for a, yi in zip(acts, y, strict=True)),
            "false_blocks": sum(a == "block" and yi == 0 for a, yi in zip(acts, y, strict=True)),
            "positives": int(y.sum()),
        }
    return out


@dataclass(frozen=True)
class Verdict:
    action: Action
    probabilities: dict[str, float]
    actions: dict[str, Action]
    shadow: bool = False
    latency_ms: float = 0.0
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def allowed(self) -> bool:
        """Whether the caller should proceed. Always true in shadow mode."""
        return self.shadow or self.action == "allow"


class Guard:
    """A forged checkpoint plus its fitted thresholds, as one production decision.

        guard = Guard("runs/agentguard/model")
        verdict = guard.check(tool_call, questions=["destructive"])
        if not verdict.allowed: ...

    Every check emits an OpenTelemetry span. `shadow=True` computes and records the verdict but
    lets everything through, for measuring a guard on live traffic before it can block anything.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        shadow: bool = False,
        on_error: Action = "escalate",
    ) -> None:
        self.model_dir = str(model_dir)
        meta = json.loads((Path(model_dir) / "rl_agent_config.json").read_text()).get("laya_forge")
        if not meta or not meta.get("policy"):
            raise ValueError(f"{model_dir} has no laya-forge policy; train it with `laya-forge run`")
        self.questions: dict[str, dict[str, Any]] = meta["questions"]
        self.policy: dict[str, dict[str, float]] = meta["policy"]
        self.agent = load_agent(self.model_dir, device)
        self.shadow = shadow
        self.on_error = on_error

    def check(self, state: State, questions: Sequence[str] | None = None) -> Verdict:
        qids = list(questions or self.policy)
        unknown = [q for q in qids if q not in self.policy]
        if unknown:
            raise ValueError(f"no policy for {unknown}; have {sorted(self.policy)}")
        with tracer.start_as_current_span("laya_forge.guard.check") as span:
            span.set_attribute("laya_forge.model", self.model_dir)
            span.set_attribute("laya_forge.questions", qids)
            span.set_attribute("laya_forge.shadow", self.shadow)
            t = time.perf_counter()
            try:
                raw = self.agent.predict(state, {q: self.questions[q] for q in qids})
            except Exception as e:  # a guard that raises takes the whole request down with it
                span.record_exception(e)
                span.set_attribute("laya_forge.action", self.on_error)
                return Verdict(
                    self.on_error,
                    {},
                    {},
                    self.shadow,
                    (time.perf_counter() - t) * 1000,
                    f"{type(e).__name__}: {e}",
                )
            ms = (time.perf_counter() - t) * 1000
            probs = {q: float(raw["answers"][q]["noul"]) for q in qids}
            acts = {q: action_for(p, self.policy[q]) for q, p in probs.items()}
            action = max(acts.values(), key=lambda a: _SEVERITY[a])
            for q, p in probs.items():
                span.set_attribute(f"laya_forge.p.{q}", p)
            span.set_attribute("laya_forge.action", action)
            span.set_attribute("laya_forge.latency_ms", ms)
            return Verdict(action, probs, acts, self.shadow, ms, None, raw)
