# Code.md — ft_linear_regression, every line defensible

Theory in Aboudi.md; this maps it to the code.

---

## 1. What's new here

- First **Python** project in the portfolio (language free; chosen for the
  visualization bonus and zero-friction CSV/JSON handling).
- First **ML pipeline**: train → save artifact → load artifact → infer, in two separate
  programs sharing only `thetas.json` and a tiny common module.
- Numerical-stability engineering (normalization, σ=0 guard) rather than syscall-level
  correctness.

## 2. File-by-file

### common.py — shared I/O and the hypothesis
- `estimate_price(mileage, t0, t1)` — the subject's hypothesis, one line, used by all
  four programs (the evaluator checks "the use of the specified hypothesis": point here).
- `THETAS_FILE` is anchored to `os.path.dirname(os.path.abspath(__file__))`, not the
  current working directory — running the programs from anywhere still finds the same
  artifact. `DEFAULT_DATA` likewise points into `data/`.
- `load_thetas()` returns (0.0, 0.0) on FileNotFoundError — that *is* the subject's
  "before training, θs are 0" requirement, implemented as the absence of the artifact.
  A *corrupted* file is a hard error instead (silently predicting nonsense would be
  worse than stopping).
- `load_dataset()` is strict on purpose: exactly 2 columns; the first line may fail
  float-parsing (header) but any later non-numeric row aborts **with the line number**;
  negative mileage/price rejected; empty dataset rejected. Defense answer for "why
  strict?": a trainer silently skipping half its data learns from a dataset you didn't
  give it.

### train.py
- `train(xs, ys, lr, iters)` is the whole algorithm:
  - μ, σ computed by hand (population σ — consistent since we also divide gradients
    by m, not m−1). `sigma == 0 → sigma = 1`: all-identical mileages then normalize to
    all-zeros, θ1 stays 0, θ0 learns the mean price — graceful degeneration (tested).
  - The loop is the subject verbatim: `errors` = estimate − price (current θs,
    normalized x), `tmp0 = lr·Σe/m`, `tmp1 = lr·Σe·x/m`, then
    **`t0, t1 = t0 - tmp0, t1 - tmp1`** — Python's tuple assignment evaluates the right
    side entirely before binding: the *simultaneous update* in one idiomatic line.
  - Return denormalized: `(t0 - t1*mu/sigma, t1/sigma)` — the algebra from
    Aboudi.md §6.5.
- Defaults lr=0.1, iters=1000: on z-scored data the optimum is reached to machine
  precision well within that; both are CLI-overridable (`--learning-rate`,
  `--iterations`) for the "what if α is too big?" defense demo — run with
  `--learning-rate 2.5` and watch it produce NaN, on purpose.
- argparse gives `-h`, typed args, and positional-dataset-with-default for free.

### predict.py
- `input()` with EOFError guard (piped/empty stdin doesn't traceback), `strip()`,
  `float()` in try/except, negative check. Output formatted `:.2f` (it's a price);
  the mileage echoes back with `:g` (no trailing zeros).
- Note what it does **not** do: read the dataset, import matplotlib, know about
  normalization. Inference is intentionally dumb — two numbers and a multiply.

### precision.py (bonus)
- Residuals via the same `estimate_price` on **raw** x (proving the saved θs live in raw
  coordinates). MAE, RMSE per Aboudi.md §6.6; R² with the SStot = 0 corner handled
  (all-equal prices: R² defined as 1 if residuals are also 0, else 0).
- Warns on stderr if θs are (0,0) — almost always "you forgot to train".

### plot.py (bonus)
- Backend chosen **before** `pyplot` import (matplotlib locks the backend at import —
  why the `# noqa: E402` comments exist): headless (no DISPLAY/WAYLAND_DISPLAY) → Agg,
  always `savefig("plot.png")`, `show()` only when interactive.
- Line drawn from just two endpoints (min/max mileage) — it's a line; two points define
  it; no linspace needed.

## 3. Design decisions (and the Y-instead question)

- **Why normalize at all?** Try it: raw km with any α that moves θ0 → NaN by iteration
  ~20. (Canyon geometry, Aboudi.md §6.5.) The alternative "use α=1e-11 and 10⁸
  iterations" technically converges θ1 but leaves θ0 barely moved — wrong line.
- **Why z-score and not min-max?** Both fix the scale; z-score also centers (μ=0), which
  *decouples* θ0 and θ1 (the gradients become independent) — faster, and the
  denormalization algebra stays two terms.
- **Why save denormalized θs instead of saving μ/σ alongside?** The artifact then plugs
  into the subject's hypothesis with no extra state; predict.py stays spec-pure. Saving
  μ/σ would work but spreads training internals into inference.
- **Why JSON and not CSV/pickle for thetas?** Human-readable at defense, schema'd keys,
  no arbitrary-code-execution risk (pickle), stdlib.
- **Why fixed iterations and not a convergence threshold?** Determinism (same input →
  same output, same runtime) and simplicity; with convex J + normalized data, 1000 iters
  overshoots convergence by a wide margin. A delta-threshold flag would be easy but adds
  a parameter to defend with no observable benefit here.
- **Why is plot.py separate from train.py?** The subject's bonus describes plotting as
  its own capability; coupling it into training would make the *mandatory* program import
  matplotlib — a heavyweight dependency the mandatory part must not need.

## 4. Libraries used (subject: any, unless they do the work)

| import | used in | what it does for us | why it's not "doing the work" |
|---|---|---|---|
| `csv` | common | splits comma-separated lines, handles quoting | parsing, not regression |
| `json` | common | (de)serializes two floats | storage, not regression |
| `os`, `sys` | all | paths anchored to the script; stderr+exit codes | plumbing |
| `argparse` | train/precision/plot | CLI flags, `-h` | UX |
| `matplotlib` | plot.py only (bonus) | draws pixels | visualization; it never sees the math — we hand it the already-fitted line's two endpoints |

The regression itself: list comprehensions, `sum()`, `zip()`, `**0.5`. Nothing imported
computes a mean, a variance, a gradient, or a fit.

## 5. Anticipated evaluator questions

- **"Show me the subject's formulas in your code."** train.py loop: four lines, named
  `tmp0`/`tmp1` to mirror the subject's notation. Hypothesis: common.py
  `estimate_price`.
- **"Is your update simultaneous?"** `t0, t1 = t0 - tmp0, t1 - tmp1` — right side fully
  evaluated first; show by adding a `print` between... there is no between. Tuple
  semantics guarantee it.
- **"The subject doesn't mention normalization."** It specifies the hypothesis and the
  update rule; both are verbatim. Training coordinates are an implementation detail and
  the saved θs satisfy the specified hypothesis on raw mileage — demonstrably:
  `precision.py` computes residuals on raw x with the saved θs.
- **"What's m?"** The number of dataset records (the subject's joke). 
- **"Make it diverge."** `python3 train.py --learning-rate 2.5` → NaN: α above the
  stability bound for a z-scored feature (the bound is α < 2 for the θ1 direction).
- **"Why does predict give 0 before training?"** No thetas.json → (0,0) by spec; the
  estimate is honestly 0.00.
- **"Untrained predict on garbage input?"** Non-numeric → error exit 1; negative →
  error exit 1; EOF → error exit 1. No tracebacks anywhere (tested).
- **"How do I know your trainer is right?"** test_flr.sh trains on a generated noiseless
  line (8000 − 0.021·km) and asserts the recovered θs match the generator to 0.1% — a
  ground-truth test no real dataset allows. R² then = 1.000000 exactly.

## 6. Re-verify

```bash
bash test_flr.sh                      # 11 checks, synthetic ground truth
# once data/data.csv exists (get_these.txt):
python3 train.py && python3 precision.py && python3 plot.py
echo 85000 | python3 predict.py
```
