#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict


def parse_float(value: str) -> float:
    if value == "" or value.lower() == "nan":
        return math.nan
    return float(value)


def parse_int(value: str) -> int:
    return int(value)


def load_rows(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "ell_level": parse_int(row["ell_level"]),
                    "iter_in_ell": parse_int(row["iter_in_ell"]),
                    "ell": parse_float(row["ell"]),
                    "min_nonzeros": parse_int(row["min_nonzeros"]),
                    "baseline_cost": parse_float(row["baseline_cost"]),
                    "trial_cost": parse_float(row["trial_cost"]),
                    "baseline_angle": parse_float(row["baseline_angle"]),
                    "trial_angle": parse_float(row["trial_angle"]),
                    "raw_dx_norm": parse_float(row["raw_dx_norm"]),
                    "accepted_dx_norm": parse_float(row["accepted_dx_norm"]),
                    "accepted": parse_int(row["accepted"]),
                    "line_search_steps": parse_int(row["line_search_steps"]),
                }
            )
    return rows


def check_monotonic(rows):
    violations = []
    accepted_rows = [r for r in rows if r["accepted"] == 1]
    for r in accepted_rows:
        if math.isfinite(r["baseline_cost"]) and math.isfinite(r["trial_cost"]):
            if r["trial_cost"] > r["baseline_cost"] + 1e-8:
                violations.append(
                    (
                        "cost",
                        r["ell_level"],
                        r["iter_in_ell"],
                        r["baseline_cost"],
                        r["trial_cost"],
                    )
                )
        if math.isfinite(r["baseline_angle"]) and math.isfinite(r["trial_angle"]):
            if r["trial_angle"] + 1e-8 < r["baseline_angle"]:
                violations.append(
                    (
                        "angle",
                        r["ell_level"],
                        r["iter_in_ell"],
                        r["baseline_angle"],
                        r["trial_angle"],
                    )
                )
    return accepted_rows, violations


def summarize(rows):
    by_ell = defaultdict(list)
    for row in rows:
        by_ell[row["ell_level"]].append(row)

    for ell_level in sorted(by_ell):
        chunk = by_ell[ell_level]
        ell = chunk[0]["ell"]
        accepted = sum(r["accepted"] for r in chunk)
        last = chunk[-1]
        finite_costs = [r["trial_cost"] for r in chunk if math.isfinite(r["trial_cost"])]
        finite_angles = [r["trial_angle"] for r in chunk if math.isfinite(r["trial_angle"])]
        min_cost = min(finite_costs) if finite_costs else math.nan
        max_angle = max(finite_angles) if finite_angles else math.nan
        print(
            f"ell_level={ell_level} ell={ell:.6g} rows={len(chunk)} "
            f"accepted={accepted} final_step={last['accepted_dx_norm']:.6g} "
            f"min_cost={min_cost:.6g} max_angle={max_angle:.6g}"
        )


def main():
    parser = argparse.ArgumentParser(description="Check monotonicity of multiframe IRLS CSV traces.")
    parser.add_argument("csv_path", help="Path to the multiframe trace CSV")
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    if not rows:
        print("empty trace")
        return 1

    accepted_rows, violations = check_monotonic(rows)
    print(f"rows={len(rows)} accepted_rows={len(accepted_rows)}")
    summarize(rows)

    if violations:
        print("violations:")
        for kind, ell_level, inner, before, after in violations[:20]:
            print(
                f"  {kind} ell_level={ell_level} iter_in_ell={inner} "
                f"before={before:.9g} after={after:.9g}"
            )
        return 2

    print("monotonicity_check=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
