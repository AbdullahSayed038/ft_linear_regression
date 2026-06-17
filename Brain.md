# Brain.md — ft_linear_regression

## Current state
- **Project:** ft_linear_regression (subject v4.1, PDF in `../Subjects/`)
- **Phase:** CODE + BONUS COMPLETE, verified on synthetic data. WAITING on official data.csv.

## Checklist
- [x] predict.py — prompts mileage, applies θ0 + θ1·x, 0/0 before training
- [x] train.py — subject's exact GD update, simultaneous θ update, z-score normalize → denormalize
- [x] Bonus: precision.py (MAE/RMSE/R²) + plot.py (scatter + line → plot.png)
- [x] test_flr.sh — 11 checks ALL PASS in WSL (noiseless recovery, error paths, σ=0 guard)
- [ ] **BLOCKED on Aboudi**: official data.csv → `data/data.csv` (see get_these.txt), then re-run suite + plot sanity check
- [ ] After data arrives: run `python3 train.py && python3 plot.py`, eyeball the fit, update this file

## Key decisions (one line each)
- Python, stdlib-only core (csv/json/math by hand) — no numpy at all, so "library does the work" is unanswerable
- Normalize mileage during training ONLY (raw km ~1e5 diverges GD); denormalize θ before saving so the SAVED θ work on raw km with the subject's hypothesis
- thetas.json next to the scripts (path from __file__, not cwd); missing file = (0,0) per subject
- lr=0.1, iters=1000 defaults (normalized data converges in a few hundred); both overridable via CLI flags
- Strict CSV: one optional header line, any other bad row = fatal error with line number; negatives rejected
- σ=0 (all-same mileage) guarded: σ:=1 → θ1=0, θ0=mean price (tested)
- matplotlib only in plot.py (bonus), Agg backend fallback when headless

## Known issues / revisit
- None failing. Trained on synthetic noiseless line: θ recovered to <0.1%. Official-data sanity still pending.
- WSL matplotlib installed via `pip3 --break-system-packages` (user site).

## Aboudi needs to provide
- `data/data.csv` (official intra attachment) — see get_these.txt

## File map
- predict.py / train.py (mandatory) · precision.py / plot.py (bonus) · common.py (shared I/O + hypothesis)
- test_flr.sh (synthetic fixtures in tests/) · docs: Aboudi.md, Code.md · subject.txt
