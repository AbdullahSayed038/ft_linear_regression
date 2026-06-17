#!/usr/bin/env python3
"""Bonus: scatter the dataset and draw the trained regression line.
Always writes plot.png; opens a window too when a display is available."""

import argparse
import os
import sys

import matplotlib

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from common import DEFAULT_DATA, estimate_price, load_dataset, \
    load_thetas  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Plot data + fitted line.")
    parser.add_argument("data", nargs="?", default=DEFAULT_DATA)
    parser.add_argument("--out", default="plot.png")
    args = parser.parse_args()

    theta0, theta1 = load_thetas()
    xs, ys = load_dataset(args.data)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, color="#2a7de1", label="dataset")
    x_lo, x_hi = min(xs), max(xs)
    line_x = [x_lo, x_hi]
    line_y = [estimate_price(x, theta0, theta1) for x in line_x]
    ax.plot(line_x, line_y, color="#e1572a", linewidth=2,
            label=f"fit: {theta0:.2f} + {theta1:.6f}*x")
    ax.set_xlabel("mileage (km)")
    ax.set_ylabel("price")
    ax.set_title("ft_linear_regression")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
