#!/usr/bin/env python3
"""Program #2: train the model with the subject's gradient-descent update
and save theta0/theta1 for predict.py.

The update rule (per iteration, simultaneous):
    tmp0 = learningRate * (1/m) * sum(estimate(x[i]) - y[i])
    tmp1 = learningRate * (1/m) * sum((estimate(x[i]) - y[i]) * x[i])
    theta0 -= tmp0 ; theta1 -= tmp1

Training runs on z-score normalized mileage (raw km ~1e5 make the raw-scale
gradient diverge for any usable learning rate); the learned thetas are then
converted back so the SAVED values plug straight into
estimatePrice(raw_mileage) = theta0 + theta1 * raw_mileage."""

import argparse
import sys

from common import DEFAULT_DATA, die, load_dataset, save_thetas


def train(xs, ys, learning_rate, iterations):
    m = len(xs)
    mu = sum(xs) / m
    variance = sum((x - mu) ** 2 for x in xs) / m
    sigma = variance ** 0.5
    if sigma == 0.0:
        sigma = 1.0
    xn = [(x - mu) / sigma for x in xs]

    t0, t1 = 0.0, 0.0
    for _ in range(iterations):
        errors = [(t0 + t1 * x) - y for x, y in zip(xn, ys)]
        tmp0 = learning_rate * sum(errors) / m
        tmp1 = learning_rate * sum(e * x for e, x in zip(errors, xn)) / m
        t0, t1 = t0 - tmp0, t1 - tmp1

    # Denormalize: t0 + t1*(x-mu)/sigma  ==  (t0 - t1*mu/sigma) + (t1/sigma)*x
    return t0 - t1 * mu / sigma, t1 / sigma


def main():
    parser = argparse.ArgumentParser(description="Train the price model.")
    parser.add_argument("data", nargs="?", default=DEFAULT_DATA,
                        help="dataset CSV (mileage,price)")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.learning_rate <= 0 or args.iterations <= 0:
        die("learning rate and iterations must be positive")

    xs, ys = load_dataset(args.data)
    theta0, theta1 = train(xs, ys, args.learning_rate, args.iterations)
    save_thetas(theta0, theta1)
    print(f"trained on {len(xs)} samples")
    print(f"theta0 = {theta0}")
    print(f"theta1 = {theta1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
