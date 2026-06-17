#!/usr/bin/env python3
"""Bonus: measure how good the trained line is on a dataset.
Reports MAE, RMSE and R^2 (coefficient of determination)."""

import argparse
import sys

from common import DEFAULT_DATA, die, estimate_price, load_dataset, \
    load_thetas


def main():
    parser = argparse.ArgumentParser(description="Model precision metrics.")
    parser.add_argument("data", nargs="?", default=DEFAULT_DATA)
    args = parser.parse_args()

    theta0, theta1 = load_thetas()
    if theta0 == 0.0 and theta1 == 0.0:
        print("warning: thetas are 0,0 (model not trained?)",
              file=sys.stderr)
    xs, ys = load_dataset(args.data)
    m = len(xs)

    residuals = [estimate_price(x, theta0, theta1) - y
                 for x, y in zip(xs, ys)]
    mae = sum(abs(r) for r in residuals) / m
    rmse = (sum(r * r for r in residuals) / m) ** 0.5
    mean_y = sum(ys) / m
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum(r * r for r in residuals)
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    print(f"samples: {m}")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
