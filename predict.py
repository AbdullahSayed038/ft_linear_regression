#!/usr/bin/env python3
"""Program #1: prompt for a mileage, print the estimated price using
estimatePrice(mileage) = theta0 + theta1 * mileage.
Before any training, theta0 = theta1 = 0 (so every estimate is 0)."""

import sys

from common import die, estimate_price, load_thetas


def main():
    theta0, theta1 = load_thetas()
    try:
        raw = input("Enter a mileage (km): ")
    except EOFError:
        die("no input")
    raw = raw.strip()
    try:
        mileage = float(raw)
    except ValueError:
        die(f"'{raw}' is not a number")
    if mileage < 0:
        die("a mileage cannot be negative")
    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price for {mileage:g} km: {price:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
