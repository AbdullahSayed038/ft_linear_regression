"""Shared helpers for ft_linear_regression: dataset / thetas I/O and the
hypothesis. Kept dependency-free on purpose (stdlib only)."""

import csv
import json
import os
import sys

THETAS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "thetas.json"
)
DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "data.csv"
)


def die(msg):
    """Print an error on stderr and exit with status 1."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def estimate_price(mileage, theta0, theta1):
    """The subject's hypothesis: estimatePrice(x) = theta0 + theta1 * x."""
    return theta0 + theta1 * mileage


def load_thetas():
    """Return (theta0, theta1). Missing file -> (0, 0), as the subject
    requires before any training has happened. Corrupted file -> error."""
    try:
        with open(THETAS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return float(data["theta0"]), float(data["theta1"])
    except FileNotFoundError:
        return 0.0, 0.0
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        die(f"corrupted thetas file {THETAS_FILE}: {exc}")


def save_thetas(theta0, theta1):
    with open(THETAS_FILE, "w", encoding="utf-8") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f, indent=2)
        f.write("\n")


def load_dataset(path):
    """Read a two-column CSV (mileage,price). The first line may be a
    header; every other line must parse as two floats. Returns (xs, ys)."""
    xs, ys = [], []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for lineno, row in enumerate(reader, start=1):
                if not row or all(cell.strip() == "" for cell in row):
                    continue
                if len(row) != 2:
                    die(f"{path}:{lineno}: expected 2 columns, got {len(row)}")
                try:
                    x, y = float(row[0]), float(row[1])
                except ValueError:
                    if lineno == 1:
                        continue
                    die(f"{path}:{lineno}: not numeric: {row}")
                if x < 0 or y < 0:
                    die(f"{path}:{lineno}: negative value: {row}")
                xs.append(x)
                ys.append(y)
    except OSError as exc:
        die(f"cannot read dataset: {exc}")
    if not xs:
        die(f"{path}: no data rows")
    return xs, ys
