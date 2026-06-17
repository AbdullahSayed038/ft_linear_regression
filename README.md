# ft_linear_regression

An introduction to **machine learning**: predict a car's price from its mileage using a
single-feature linear regression trained with **gradient descent** — implemented in pure
Python with no NumPy, so the maths is all visible.

> 42 ML-intro project. Code + bonus complete, verified on synthetic data; awaiting the
> official `data.csv`.

## The two programs

- **`predict.py`** — prompts for a mileage and returns the estimated price using the
  hypothesis `price = θ₀ + θ₁ · mileage`. Before any training it predicts 0 (θ = 0, 0).
- **`train.py`** — runs the subject's exact gradient-descent update with **simultaneous**
  θ₀/θ₁ updates, then saves the learned parameters to `thetas.json`.

### Bonus

- `precision.py` — reports MAE, RMSE, and R² of the trained model.
- `plot.py` — scatter of the data with the fitted line, saved to `plot.png`.

## Why normalisation matters

Raw mileage (~10⁵ km) makes gradient descent diverge, so `train.py` **z-score
normalises** the mileage during training only, then **denormalises** θ before saving —
so the stored parameters work directly on raw kilometres with the subject's hypothesis.
This is the key conceptual hurdle of the project, and it's handled explicitly.

Defaults: learning rate 0.1, 1000 iterations (both overridable via CLI). The all-same
mileage case (σ = 0) is guarded so it can't divide by zero.

## Build & run

```sh
python3 train.py            # reads data/data.csv -> thetas.json
python3 predict.py          # prompts for a mileage, prints the estimate
python3 precision.py        # bonus: MAE / RMSE / R²
python3 plot.py             # bonus: scatter + fitted line -> plot.png
```

## Verification

```sh
bash test_flr.sh     # 11 checks, all pass
```

Covers noiseless recovery (θ recovered to < 0.1% on a synthetic line), error paths
(malformed CSV with line numbers, negative values), and the σ = 0 guard.

## Required data (not in the repo)

The official `data.csv` goes in `data/` — see `get_these.txt`. Then re-run `train.py`
and `plot.py` and eyeball the fit.

## Notes for defense

- **Stdlib only** (csv / json / math by hand, no NumPy) — so "the library did the work"
  is not answerable.
- `thetas.json` lives next to the scripts (path from `__file__`, not the cwd); a missing
  file means θ = (0, 0), per the subject.

## Status

Code + bonus complete, verified on synthetic data. Blocked only on the official dataset
for the real-data sanity check.
