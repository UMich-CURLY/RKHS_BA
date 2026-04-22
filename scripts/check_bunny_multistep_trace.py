#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def gt_pose():
    ang = math.radians(30.0)
    cy, sy = math.cos(ang), math.sin(ang)
    return [
        [cy, 0.0, sy, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transpose3(r):
    return [[r[j][i] for j in range(3)] for i in range(3)]


def matmul4(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def invert_pose(T):
    R = [row[:3] for row in T[:3]]
    t = [T[0][3], T[1][3], T[2][3]]
    Rt = transpose3(R)
    tinv = [-sum(Rt[i][j] * t[j] for j in range(3)) for i in range(3)]
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            out[i][j] = Rt[i][j]
        out[i][3] = tinv[i]
    out[3][3] = 1.0
    return out


def dist_se3(T_err):
    R = [row[:3] for row in T_err[:3]]
    t = [T_err[0][3], T_err[1][3], T_err[2][3]]
    tr = R[0][0] + R[1][1] + R[2][2]
    angle = math.acos(clamp((tr - 1.0) * 0.5, -1.0, 1.0))
    trans = math.sqrt(sum(v * v for v in t))
    return math.sqrt(angle * angle + trans * trans)


def load_obj_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "ell_level": int(row["ell_level"]),
                    "iter_in_ell": int(row["iter_in_ell"]),
                    "ell": float(row["ell"]),
                    "baseline_cost": float(row["baseline_cost"]) if row["baseline_cost"] not in ("", "nan") else math.nan,
                    "trial_cost": float(row["trial_cost"]) if row["trial_cost"] not in ("", "nan") else math.nan,
                    "baseline_angle": float(row["baseline_angle"]) if row["baseline_angle"] not in ("", "nan") else math.nan,
                    "trial_angle": float(row["trial_angle"]) if row["trial_angle"] not in ("", "nan") else math.nan,
                    "accepted": int(row["accepted"]),
                }
            )
    return [r for r in rows if r["accepted"] == 1]


def load_pose_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["frame_id"] != "1":
                continue
            T = [
                [float(row["r00"]), float(row["r01"]), float(row["r02"]), float(row["t0"])],
                [float(row["r10"]), float(row["r11"]), float(row["r12"]), float(row["t1"])],
                [float(row["r20"]), float(row["r21"]), float(row["r22"]), float(row["t2"])],
                [0.0, 0.0, 0.0, 1.0],
            ]
            err = matmul4(invert_pose(T), gt_pose())
            rows.append(
                {
                    "ell_level": int(row["ell_level"]),
                    "iter_in_ell": int(row["iter_in_ell"]),
                    "ell": float(row["ell"]),
                    "gt_error": dist_se3(err),
                }
            )
    return rows


def monotone_same_ell(rows, before_key, after_key, sign):
    violations = []
    for row in rows:
        before = row[before_key]
        after = row[after_key]
        if not math.isfinite(before) or not math.isfinite(after):
            continue
        ok = after + 1e-8 >= before if sign == "up" else after <= before + 1e-8
        if not ok:
            violations.append((row["ell_level"], row["iter_in_ell"], before, after))
    return violations


def summarize_gt(pose_rows):
    by_ell = defaultdict(list)
    for row in pose_rows:
        by_ell[row["ell_level"]].append(row)
    for ell_level in sorted(by_ell):
        chunk = by_ell[ell_level]
        print(
            f"gt ell_level={ell_level} ell={chunk[0]['ell']:.6g} "
            f"start={chunk[0]['gt_error']:.9g} best={min(r['gt_error'] for r in chunk):.9g} "
            f"end={chunk[-1]['gt_error']:.9g}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-step bunny trend logs.")
    parser.add_argument("objective_csv")
    parser.add_argument("pose_csv")
    args = parser.parse_args()

    obj_rows = load_obj_rows(args.objective_csv)
    pose_rows = load_pose_rows(args.pose_csv)
    if not obj_rows or not pose_rows:
        print("missing accepted rows")
        return 1

    angle_viol = monotone_same_ell(obj_rows, "baseline_angle", "trial_angle", "up")
    cost_viol = monotone_same_ell(obj_rows, "baseline_cost", "trial_cost", "down")

    print(f"accepted_rows={len(obj_rows)} pose_rows={len(pose_rows)}")
    print(f"same_ell_angle_monotone={'PASS' if not angle_viol else 'FAIL'}")
    print(f"same_ell_cost_monotone={'PASS' if not cost_viol else 'FAIL'}")
    summarize_gt(pose_rows)
    best = min(pose_rows, key=lambda r: r["gt_error"])
    last = pose_rows[-1]
    print(f"gt best ell_level={best['ell_level']} iter={best['iter_in_ell']} err={best['gt_error']:.9g}")
    print(f"gt last ell_level={last['ell_level']} iter={last['iter_in_ell']} err={last['gt_error']:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
