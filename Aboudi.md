# Aboudi.md — ft_linear_regression, taught from zero

---

## 1. Project summary — what we built

Two programs that together form your first machine-learning system:

- **train.py** reads a CSV of (mileage, price) pairs and *learns* two numbers, θ0 and θ1,
  such that the line `price = θ0 + θ1·mileage` fits the data as well as possible. The
  learning happens by **gradient descent**, using exactly the update formulas the subject
  prints. The learned θs are saved to `thetas.json`.
- **predict.py** asks for a mileage and answers with `θ0 + θ1·mileage`. Before any
  training both θs are 0, so it answers 0 — that's specified behavior, not a bug.

Bonus: **plot.py** (see the data cloud and the fitted line) and **precision.py**
(MAE, RMSE, R² — numbers that say *how good* the fit is).

## 2. What this project teaches

- **The shape of all supervised ML**: model (hypothesis) → cost (how wrong am I?) →
  optimizer (how do I get less wrong?). Every neural network you'll ever train — including
  the Multilayer Perceptron project later — is this exact triangle, scaled up.
- **Gradient descent** — the single most important algorithm in modern ML. GPT models are
  trained with a descendant of the two-line update rule you implement here.
- **Feature scaling** — why raw data often *cannot* be optimized directly, and what
  normalization actually does geometrically.
- **The train/inference split** — training is expensive and happens once; inference is
  cheap and reads the saved parameters. `thetas.json` is your first model artifact, the
  ancestor of every `.safetensors` file.
- **Evaluation honesty** — R² and friends: measuring your model instead of vibing it.

## 3. What people usually find hard

- **Divergence on raw data (the wall everyone hits).** Mileages are ~100,000; prices are
  ~5,000. With raw x, the θ1 gradient is scaled by x ~ 1e5, so any learning rate big
  enough to move θ0 makes θ1 oscillate and explode to NaN within iterations. People
  conclude "my formula is wrong" — the formula is fine, the *scale* is wrong.
  Fix: normalize x for training, denormalize the θs after (§6.5).
- **Sequential instead of simultaneous update.** If you update θ0 first and then use the
  *new* θ0 while computing θ1's gradient, you're descending a different (wrong) direction.
  The subject warns about this explicitly. Compute both tmp values from the same θs, then
  assign both.
- **Sign confusion** — the formulas use (estimate − price), and the update *subtracts*
  tmp. Flip either one and your line runs away from the data instead of toward it.
- **Confusing the two programs' θs.** estimatePrice inside training uses the *current,
  in-progress* θs on *normalized* x; predict.py uses the *final, denormalized* θs on raw
  x. Same formula, different coordinate systems.
- **"My R² is only 0.73, my code is broken."** No — real data has noise; a perfect line
  through a noisy cloud doesn't exist. R² ≈ 1 happens only on synthetic noiseless data
  (which is exactly how our test suite proves correctness).

## 4. The challenges we faced

- **No official dataset in the repo** — data.csv is an intra attachment. Rather than fake
  it, the suite generates *labeled synthetic fixtures* (a noiseless line), which is
  actually the stronger test: on noiseless data the recovered θs must match the generating
  line to <0.1%, which is a pass/fail truth no real dataset gives you. get_these.txt asks
  for the real file for the final sanity pass.
- **Denormalization algebra** — expanding θ0' + θ1'·(x−μ)/σ and collecting terms gives
  θ0 = θ0' − θ1'μ/σ and θ1 = θ1'/σ. One sign slip here and predictions are garbage while
  training "looks" converged. Verified by the noiseless-recovery test.
- **Zero-variance dataset** (every car has the same mileage): σ = 0 → division by zero.
  Guard σ := 1; gradient for θ1 is then 0, the model degenerates gracefully to
  "predict the mean price" — tested.
- **Headless plotting** — WSL has no display by default; plot.py picks the Agg backend
  and writes plot.png instead of crashing on `plt.show()`.

## 5. Real-world relevance

- **This is the "hello world" of a trillion-dollar industry.** Price estimation from
  features is literally what Zillow/Kelley Blue Book/insurance pricing teams do, with more
  features and fancier models — same triangle: hypothesis, cost, optimizer.
- **Gradient descent is *the* workhorse**: logistic regression (your DSLR project), neural
  nets (MLP project), deep learning at every scale. Adam, RMSProp, momentum — all are
  decorated versions of `θ -= lr * gradient`.
- **Feature scaling is everyday practice** — sklearn's StandardScaler exists because every
  practitioner hits the exact divergence you hit here.
- **Model artifacts and serving** — saving learned parameters, loading them in a separate
  cheap process: that's the architecture of every ML deployment, from thetas.json to
  TensorFlow Serving.

## 6. The subject from zero

### 6.1 The problem: regression

You have m examples (x⁽ⁱ⁾, y⁽ⁱ⁾) — mileage and price of sold cars. You want a function
that maps any mileage to a predicted price. Choosing the family of functions is the
**model choice**; the subject chooses the simplest useful one — a straight line:

    estimatePrice(x) = θ0 + θ1·x

θ0 (the intercept: price of a 0-km car) and θ1 (the slope: € lost per km, will come out
negative) are the **parameters**. "Learning" = picking the parameter values that fit the
data best. With one feature (mileage), this is **simple (univariate) linear regression**.

### 6.2 "Best fit" needs a number: the cost function

Define the error on example i: e⁽ⁱ⁾ = estimatePrice(x⁽ⁱ⁾) − y⁽ⁱ⁾. Aggregate them as the
**mean squared error**:

    J(θ0, θ1) = (1/2m) · Σ (e⁽ⁱ⁾)²

Why squared? Penalizes big misses much more than small ones, treats + and − errors
symmetrically, and — decisive for us — it's smooth, with a single global minimum (a bowl).
J is a function of the *parameters*: the data is fixed; you're moving the line.

### 6.3 Gradient descent: rolling down the bowl

The **gradient** of J is the vector of its partial derivatives — it points in the
direction of steepest *increase*. So step the opposite way, repeatedly:

    θ := θ − α · ∂J/∂θ

α is the **learning rate**: too small = crawling; too big = overshooting the valley and
diverging. Differentiating J gives exactly the subject's formulas:

    ∂J/∂θ0 = (1/m) Σ e⁽ⁱ⁾            → tmp0 = α · (1/m) Σ e⁽ⁱ⁾
    ∂J/∂θ1 = (1/m) Σ e⁽ⁱ⁾·x⁽ⁱ⁾       → tmp1 = α · (1/m) Σ e⁽ⁱ⁾·x⁽ⁱ⁾

(The ½ in J exists to cancel the 2 from differentiating the square — cosmetic.)
**Simultaneous update**: both tmp values come from the same current θs; updating θ0
mid-computation would mean evaluating the two partials at *different points* — no longer
the gradient of anything.

This is **batch** gradient descent — every step looks at all m examples. With m ≈ 24
that's free; with m = millions you'd sample mini-batches (that's SGD — same idea).

### 6.4 Why it converges here

For linear regression, J is **convex** (a paraboloid): no local minima, no saddle traps —
any learning rate below the stability threshold reaches *the* global minimum. This is why
linear regression is the perfect sandbox: when something explodes, it's your scaling or
your signs, never "stuck in a local minimum".

### 6.5 Normalization — the heart of the practical difficulty

The θ1 gradient is scaled by x (~1e5) and the θ0 gradient isn't (~1). The bowl is a
canyon: insanely steep along θ1, nearly flat along θ0. One α must serve both directions —
impossible: α small enough for θ1's direction barely moves θ0; α big enough for θ0
catapults θ1 out of the canyon → NaN.

**Z-score normalization** fixes the geometry: x' = (x − μ)/σ gives mileage mean 0 and
standard deviation 1 — same scale as the bowl's other axis. The canyon becomes a round
bowl; α = 0.1 converges in a few hundred iterations.

We train θ0', θ1' against x'. But predict.py must use *raw* kilometers, so expand:

    θ0' + θ1'·(x−μ)/σ = (θ0' − θ1'·μ/σ) + (θ1'/σ)·x
                          └──── θ0 ────┘   └─ θ1 ─┘

Save θ0, θ1 — the hypothesis in the predictor is the subject's, untouched, on raw input.
Normalization is an internal training trick, invisible from outside. (That's also the
defense answer to "the subject doesn't mention normalization": the subject specifies the
hypothesis and the update rule — both are implemented verbatim; coordinates during
training are an implementation choice, and the saved artifact honors the spec.)

### 6.6 Measuring the fit (bonus)

- **MAE** = mean |error| — "off by X euros on average", robust, intuitive.
- **RMSE** = √(mean error²) — same units as price, punishes outliers harder; this is the
  metric your cost function actually optimizes (up to the √).
- **R²** = 1 − SSres/SStot: the fraction of price variance the line explains. 1 = perfect,
  0 = no better than always answering the mean, negative = worse than the mean.
  SStot = Σ(y − ȳ)², SSres = Σ e². On the official car data expect ~0.73 — that's a good
  fit for noisy human pricing.

### 6.7 Why no numpy

The subject bans libraries that "do the work" (polyfit literally returns θ0, θ1 in closed
form — the normal equation — skipping the learning). We went further: the math is plain
Python sums and list comprehensions. Nothing to argue about at defense, and m = 24 needs
no vectorization.

### 6.8 Hold yourself to these

1. Derive ∂J/∂θ1 from J by the chain rule, on paper.
2. Why does raw-mileage training diverge? Draw the canyon.
3. Do the denormalization algebra yourself and check it against train.py.
4. What changes if you initialize θs randomly instead of 0? (For convex J: nothing but
   the path.)
5. Why must the update be simultaneous? What is the sequential version actually doing?
6. Your R² is 0.73 on real data. Is that good? What would −0.2 mean?
7. predict.py on a 1,000,000-km mileage returns a negative price. Defend or fix.
   (Defend: the model is linear by spec; extrapolation beyond data is the caller's sin.)
