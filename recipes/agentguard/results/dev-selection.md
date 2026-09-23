# laya-forge report (dev selection)

> Model-selection run, kept for the record. Calibration here used a *generated* split, since removed:
> the fitted block threshold landed at 0.024 and produced 10 false blocks on dev.
> The final recipe calibrates on hand-written data instead. Final numbers: [holdout.md](holdout.md).


**Gate: FAIL**

- dev_destructive: ECE 0.118 > 0.1

## Test set `dev_destructive` (n=63)

| Model | Accuracy [95% CI] | ECE | Brier | NLL |
|---|---|---|---|---|
| Laya as shipped | 0.746 [0.63, 0.84] | 0.191 | 0.404 | 0.612 |
| Laya, calibrated only | 0.746 [0.63, 0.84] | 0.099 | 0.393 | 0.578 |
| fine-tuned | 0.857 [0.75, 0.92] | 0.141 | 0.275 | 0.750 |
| fine-tuned + calibrated | 0.857 [0.75, 0.92] | 0.118 | 0.247 | 0.445 |

| Question | n | Laya as shipped | fine-tuned + calibrated |
|---|---|---|---|
| `destructive` | 63 | 0.746 [0.63, 0.84] | 0.857 [0.75, 0.92] |

| Question | allow | escalate | block | missed positives | false blocks |
|---|---|---|---|---|---|
| `destructive` | 37% | 0% | 63% | 1/31 | 10 |

## Test set `dev_injection` (n=60)

| Model | Accuracy [95% CI] | ECE | Brier | NLL |
|---|---|---|---|---|
| Laya as shipped | 0.700 [0.57, 0.80] | 0.073 | 0.409 | 0.598 |
| Laya, calibrated only | 0.700 [0.57, 0.80] | 0.057 | 0.404 | 0.588 |
| fine-tuned | 0.883 [0.78, 0.94] | 0.105 | 0.214 | 0.865 |
| fine-tuned + calibrated | 0.883 [0.78, 0.94] | 0.090 | 0.200 | 0.505 |

| Question | n | Laya as shipped | fine-tuned + calibrated |
|---|---|---|---|
| `injection` | 60 | 0.700 [0.57, 0.80] | 0.883 [0.78, 0.94] |

| Question | allow | escalate | block | missed positives | false blocks |
|---|---|---|---|---|---|
| `injection` | 45% | 0% | 55% | 2/30 | 5 |

## Thresholds (fitted on the calibration split)

| Question | allow below | block at |
|---|---|---|
| `destructive` | 0.024 | 0.024 |
| `injection` | 0.064 | 0.064 |

## Run

- base model: `convaiinnovations/laya` on `mps`
- training: 391.6s, loss per epoch [0.4289, 0.1558, 0.0583]
- temperatures: {'noul:2': 1.82}
- data (examples, sha256): train 1600 `9a92293234bc`, calibration 400 `5b53ae43f7b2`, test/dev_destructive 63 `67b2b93238fa`, test/dev_injection 60 `8e836900f362`
