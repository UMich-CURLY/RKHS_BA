#!/usr/bin/env python3

import argparse
import csv
import math


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def matmul4(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def transpose3(r):
    return [[r[j][i] for j in range(3)] for i in range(3)]


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


def gt_pose():
    ang = math.radians(30.0)
    cy, sy = math.cos(ang), math.sin(ang)
    T = [[0.0] * 4 for _ in range(4)]
    T[0] = [cy, 0.0, sy, 0.5]
    T[1] = [0.0, 1.0, 0.0, 0.0]
    T[2] = [-sy, 0.0, cy, 0.0]
    T[3] = [0.0, 0.0, 0.0, 1.0]
    return T


def main():
    parser = argparse.ArgumentParser(description="Check bunny pose-trace error against the fixed 2-frame GT.")
    parser.add_argument("csv_path")
    args = parser.parse_args()

    rows = []
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["frame_id"]) != 1:
                continue
            T = [[0.0] * 4 for _ in range(4)]
            T[0] = [float(row["r00"]), float(row["r01"]), float(row["r02"]), float(row["t0"])]
            T[1] = [float(row["r10"]), float(row["r11"]), float(row["r12"]), float(row["t1"])]
            T[2] = [float(row["r20"]), float(row["r21"]), float(row["r22"]), float(row["t2"])]
            T[3] = [0.0, 0.0, 0.0, 1.0]
            err = matmul4(invert_pose(T), gt_pose())
            rows.append(
                (
                    int(row["ell_level"]),
                    int(row["iter_in_ell"]),
                    float(row["ell"]),
                    dist_se3(err),
                    (T[0][3], T[1][3], T[2][3]),
                )
            )

    if not rows:
        print("no frame-1 rows found")
        return 1

    best = min(rows, key=lambda r: r[3])
    last = rows[-1]
    print(f"rows={len(rows)}")
    print(f"best ell_level={best[0]} iter={best[1]} ell={best[2]:.6g} pose_err={best[3]:.9g} t={best[4]}")
    print(f"last ell_level={last[0]} iter={last[1]} ell={last[2]:.6g} pose_err={last[3]:.9g} t={last[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
