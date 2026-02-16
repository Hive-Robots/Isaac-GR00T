#!/usr/bin/env python3
"""
Plot diagnostics CSV produced by eval_g1.py.

Usage:
  python gr00t/eval/real_robot/g1/robot_sdk/utils/plot_diagnostics.py
  python gr00t/eval/real_robot/g1/robot_sdk/utils/plot_diagnostics.py --csv eval_g1_diagnostics_20260216_165309.csv
  python gr00t/eval/real_robot/g1/robot_sdk/utils/plot_diagnostics.py --csv <file> --out <png> --show
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
from typing import Dict, List


DEFAULT_PATTERN = "eval_g1_diagnostics_*.csv"
REQUIRED_COLUMNS = [
    "time_since_start_s",
    "loop_elapsed_ms",
    "sleep_time_ms",
    "policy_infer_ms",
    "action_delta_l2",
    "tracking_error_l2",
    "arm_dq_l2",
    "missing_camera",
]
OPTIONAL_COLUMNS = [
    "using_hold_last",
]


def _find_latest_csv(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No diagnostics CSV found with pattern: {pattern}")
    return Path(matches[-1])


def _load_csv(path: Path) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {k: [] for k in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [k for k in REQUIRED_COLUMNS if k not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        for row in reader:
            for k in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
                raw = row.get(k, "")
                cols[k].append(float(raw) if raw not in (None, "") else float("nan"))
    return cols


def _default_out_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_plot.png")


def plot_diagnostics(data: Dict[str, List[float]], csv_path: Path, out_path: Path, show: bool) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with: uv add matplotlib"
        ) from exc

    t = data["time_since_start_s"]
    loop_ms = data["loop_elapsed_ms"]
    sleep_ms = data["sleep_time_ms"]
    infer_ms = data["policy_infer_ms"]
    action_delta = data["action_delta_l2"]
    tracking = data["tracking_error_l2"]
    arm_dq = data["arm_dq_l2"]
    missing_cam = data["missing_camera"]
    hold_last = data.get("using_hold_last", [])

    # Similar to open_loop_eval: mark inference points.
    inference_idx = [i for i, v in enumerate(infer_ms) if math.isfinite(v) and v > 0]

    # Similar visual structure to open_loop_eval.plot_trajectory_results:
    # stacked panels, global title, per-panel legend, and red inference markers.
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 20))
    fig.suptitle(f"Diagnostics - {csv_path.name}", fontsize=16, color="blue")

    axes[0].plot(t, loop_ms, label="loop_elapsed_ms", linewidth=1.2)
    axes[0].plot(t, sleep_ms, label="sleep_time_ms", linewidth=1.2)
    for j in inference_idx:
        if j == inference_idx[0]:
            axes[0].plot(t[j], loop_ms[j], "ro", label="inference point")
        else:
            axes[0].plot(t[j], loop_ms[j], "ro")
    axes[0].set_title("Loop Timing")
    axes[0].set_ylabel("ms")
    axes[0].legend()

    axes[1].plot(t, infer_ms, label="policy_infer_ms", color="tab:orange", linewidth=1.2)
    finite_infer = [v for v in infer_ms if math.isfinite(v)]
    cam_scale = max(finite_infer) if finite_infer else 1.0
    missing_cam_scaled = [v * max(cam_scale, 1.0) for v in missing_cam]
    axes[1].plot(t, missing_cam_scaled, label="missing_camera", color="tab:red", alpha=0.5)
    for j in inference_idx:
        if j == inference_idx[0]:
            axes[1].plot(t[j], infer_ms[j], "ro", label="inference point")
        else:
            axes[1].plot(t[j], infer_ms[j], "ro")
    axes[1].set_title("Policy Latency")
    axes[1].set_ylabel("ms / flag")
    axes[1].legend()

    axes[2].plot(t, action_delta, label="action_delta_l2", linewidth=1.2)
    axes[2].plot(t, tracking, label="tracking_error_l2", linewidth=1.2)
    for j in inference_idx:
        if j == inference_idx[0]:
            axes[2].plot(t[j], action_delta[j], "ro", label="inference point")
        else:
            axes[2].plot(t[j], action_delta[j], "ro")
    axes[2].set_title("Action/Tracking Error")
    axes[2].set_ylabel("L2")
    axes[2].legend()

    axes[3].plot(t, arm_dq, label="arm_dq_l2", color="tab:green", linewidth=1.2)
    for j in inference_idx:
        if j == inference_idx[0]:
            axes[3].plot(t[j], arm_dq[j], "ro", label="inference point")
        else:
            axes[3].plot(t[j], arm_dq[j], "ro")
    axes[3].set_title("Arm Velocity Norm")
    axes[3].set_ylabel("L2")
    axes[3].legend()

    if hold_last and any(math.isfinite(v) for v in hold_last):
        hold_scale = 1.0
        hold_vals = [v * hold_scale for v in hold_last]
        axes[4].plot(t, hold_vals, label="using_hold_last", color="tab:purple", linewidth=1.2)
        axes[4].set_ylim(-0.1, 1.1)
    else:
        zeros = [0.0 for _ in t]
        axes[4].plot(t, zeros, label="using_hold_last (not in csv)", color="tab:gray", linewidth=1.2)
        axes[4].set_ylim(-0.1, 1.1)
    for j in inference_idx:
        if j == inference_idx[0]:
            axes[4].plot(t[j], 1.0, "ro", label="inference point")
        else:
            axes[4].plot(t[j], 1.0, "ro")
    axes[4].set_title("Hold-Last Fallback")
    axes[4].set_ylabel("flag")
    axes[4].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[4].set_xlabel("time_since_start_s")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")
    if show:
        try:
            plt.show()
        except Exception as exc:
            print(f"Skipping interactive show: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval_g1 diagnostics CSV.")
    parser.add_argument("--csv", type=str, default=None, help="Path to diagnostics CSV.")
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help=f"Glob pattern when --csv is not set (default: {DEFAULT_PATTERN}).",
    )
    parser.add_argument("--out", type=str, default=None, help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Also display the figure interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv) if args.csv else _find_latest_csv(args.pattern)
    out_path = Path(args.out) if args.out else _default_out_path(csv_path)
    data = _load_csv(csv_path)
    try:
        plot_diagnostics(data, csv_path, out_path, show=args.show)
    except ModuleNotFoundError as exc:
        print(str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
