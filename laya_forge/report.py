"""Markdown report for a run: the numbers a reviewer needs, with intervals, and nothing else."""

from __future__ import annotations

from typing import Any

LABELS = {
    "base": "Laya as shipped",
    "base_calibrated": "Laya, calibrated only",
    "forged_raw": "fine-tuned",
    "forged": "fine-tuned + calibrated",
}


def _acc(m: dict[str, Any]) -> str:
    lo, hi = m["accuracy_ci"]
    return f"{m['accuracy']:.3f} [{lo:.2f}, {hi:.2f}]"


def render(r: dict[str, Any]) -> str:
    out = ["# laya-forge report", ""]
    gate = r.get("gate")
    if gate:
        out += [f"**Gate: {'PASS' if gate['passed'] else 'FAIL'}**", ""]
        out += [f"- {f}" for f in gate["failures"]] + ([""] if gate["failures"] else [])

    for name, res in r["test"].items():
        n = next(iter(res.values()))["overall"]["n"]
        out += [
            f"## Test set `{name}` (n={n})",
            "",
            "| Model | Accuracy [95% CI] | ECE | Brier | NLL |",
            "|---|---|---|---|---|",
        ]
        for key, label in LABELS.items():
            if key in res:
                m = res[key]["overall"]
                out.append(f"| {label} | {_acc(m)} | {m['ece']:.3f} | {m['brier']:.3f} | {m['nll']:.3f} |")
        out.append("")
        if "forged" in res and "base" in res:
            out += ["| Question | n | Laya as shipped | fine-tuned + calibrated |", "|---|---|---|---|"]
            for q, m in res["forged"]["questions"].items():
                out.append(f"| `{q}` | {m['n']} | {_acc(res['base']['questions'][q])} | {_acc(m)} |")
            out.append("")
        if res.get("policy"):
            out += [
                "| Question | allow | escalate | block | missed positives | false blocks |",
                "|---|---|---|---|---|---|",
            ]
            for q, p in res["policy"].items():
                out.append(
                    f"| `{q}` | {p['allow']:.0%} | {p['escalate']:.0%} | {p['block']:.0%} "
                    f"| {p['missed_positives']}/{p['positives']} | {p['false_blocks']} |"
                )
            out.append("")

    if "policy" in r:
        out += [
            "## Thresholds (fitted on the calibration split)",
            "",
            "| Question | allow below | block at |",
            "|---|---|---|",
        ]
        for q, th in r["policy"].items():
            block = "never" if th["block_at"] > 1 else f"{th['block_at']:.3f}"
            out.append(f"| `{q}` | {th['allow_below']:.3f} | {block} |")
        out.append("")
    if "train_seconds" in r:
        out += [
            "## Run",
            "",
            f"- base model: `{r['base_model']}` on `{r['device']}`",
            f"- training: {r['train_seconds']}s, loss per epoch {[round(x, 4) for x in r['train_loss']]}",
            f"- temperatures: {r['temperatures']}",
            "- data (examples, sha256): " + ", ".join(f"{k} {v[0]} `{v[1]}`" for k, v in r["data"].items()),
            "",
        ]
    return "\n".join(out)
