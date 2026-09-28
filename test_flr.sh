#!/bin/bash
# ft_linear_regression test suite (mandatory part only) -- run from the
# project root. Uses a SYNTHETIC fixture (tests/synth.csv).
set -u
cd "$(dirname "$0")" || exit 1
fail=0
ok() { printf 'OK  %s\n' "$*"; }
ko() { printf 'KO  %s\n' "$*"; fail=1; }

mkdir -p tests
# Noiseless line: price = 8000 - 0.021 * km  -> training must recover it.
python3 - <<'EOF'
rows = [(km, 8000 - 0.021 * km) for km in range(10000, 250001, 10000)]
with open("tests/synth.csv", "w") as f:
    f.write("km,price\n")
    for km, p in rows:
        f.write(f"{km},{p}\n")
EOF

echo "== untrained predictor returns 0"
rm -f thetas.json
out=$(echo 50000 | python3 predict.py)
echo "$out" | grep -q '0.00' && ok "predicts 0.00 before training" \
    || ko "expected 0.00, got: $out"

echo "== training recovers a noiseless line"
python3 train.py tests/synth.csv > /dev/null || ko "train.py failed"
python3 - <<'EOF' && ok "thetas within 0.1% of truth" || exit_code=1
import json
d = json.load(open("thetas.json"))
t0, t1 = d["theta0"], d["theta1"]
assert abs(t0 - 8000) / 8000 < 1e-3, f"theta0={t0}"
assert abs(t1 - (-0.021)) / 0.021 < 1e-3, f"theta1={t1}"
EOF
[ $? -ne 0 ] && ko "thetas off: $(cat thetas.json)"

echo "== predictor uses the trained thetas"
out=$(echo 100000 | python3 predict.py)
# 8000 - 0.021*100000 = 5900
echo "$out" | grep -qE '59[0-9][0-9]\.|5899\.|5900\.' \
    && ok "predict(100000) ~ 5900: $out" || ko "unexpected: $out"

echo "== error handling"
echo abc | python3 predict.py >/dev/null 2>&1 && ko "non-numeric accepted" \
    || ok "non-numeric input rejected"
echo -- -5 | python3 predict.py >/dev/null 2>&1 && ko "negative accepted" \
    || ok "negative mileage rejected"
python3 train.py /nonexistent.csv >/dev/null 2>&1 && ko "missing file ok?" \
    || ok "missing dataset rejected"
printf 'km,price\n1,2,3\n' > tests/bad.csv
python3 train.py tests/bad.csv >/dev/null 2>&1 && ko "bad row accepted" \
    || ok "malformed row rejected"
printf 'km,price\n' > tests/empty.csv
python3 train.py tests/empty.csv >/dev/null 2>&1 && ko "empty accepted" \
    || ok "empty dataset rejected"
printf '10000,5000\n10000,7000\n' > tests/flat.csv
python3 train.py tests/flat.csv >/dev/null 2>&1 \
    && ok "zero-variance dataset survives (sigma guard)" \
    || ko "zero-variance dataset crashed"

rm -f thetas.json
exit $fail
