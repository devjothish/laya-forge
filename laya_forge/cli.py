"""Command line: `laya-forge run CONFIG` and `laya-forge eval MODEL CONFIG`. Exit 1 when the gate fails."""

from __future__ import annotations

import argparse
import json
import sys

from . import pipeline
from .report import render


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="laya-forge", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fine-tune, calibrate, fit thresholds, export and gate")
    r.add_argument("config")
    e = sub.add_parser("eval", help="score a checkpoint against the base model and gate it")
    e.add_argument("model")
    e.add_argument("config")
    e.add_argument("--json", action="store_true", help="print the full results as JSON")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        results, failures = pipeline.run(args.config)
    else:
        results, failures = pipeline.evaluate(args.model, args.config)
        print(json.dumps(results, indent=2, default=float) if args.json else render(results))
    for f in failures:
        print(f"GATE FAIL: {f}", file=sys.stderr)
    print("gate:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
