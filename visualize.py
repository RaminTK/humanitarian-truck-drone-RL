"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from instance_generator import ProblemInstance


def plot_metric_bars(
    summary: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    if summary.empty or metric not in summary.columns:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = summary.pivot(index="size", columns="method", values=metric).sort_index()
    ax = pivot.plot(kind="bar", figsize=(10, 5), width=0.82)
    ax.set_title(title)
    ax.set_xlabel("Instance size")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_route(
    instance: ProblemInstance,
    trace: list[dict[str, Any]],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    truck_nodes = instance.truck_accessible
    drone_nodes = ~truck_nodes

    ax.scatter(
        instance.coords[truck_nodes, 0],
        instance.coords[truck_nodes, 1],
        c="#1f77b4",
        s=46,
        label="Truck-accessible",
    )
    ax.scatter(
        instance.coords[drone_nodes, 0],
        instance.coords[drone_nodes, 1],
        c="#d95f02",
        marker="^",
        s=56,
        label="Drone-only",
    )
    ax.scatter(
        [instance.depot[0]],
        [instance.depot[1]],
        c="black",
        marker="*",
        s=180,
        label="Depot",
    )

    for node_index, (x_coord, y_coord) in enumerate(instance.coords):
        ax.text(x_coord + 0.8, y_coord + 0.8, str(node_index), fontsize=7)

    for step in trace:
        from_x = float(step["from_x"])
        from_y = float(step["from_y"])
        to_x = float(step["to_x"])
        to_y = float(step["to_y"])
        if step["mode"] == "truck":
            ax.plot([from_x, to_x], [from_y, to_y], color="#1f77b4", linewidth=2.0, alpha=0.85)
        elif step["mode"] == "drone":
            ax.plot(
                [from_x, to_x],
                [from_y, to_y],
                color="#d95f02",
                linewidth=1.5,
                linestyle="--",
                alpha=0.75,
            )

    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_learning_curve(log_dir: str | Path, output_path: str | Path, title: str) -> None:
    rewards = _load_monitor_rewards(log_dir)
    if rewards.empty:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rewards = rewards.sort_values("t").reset_index(drop=True)
    rewards["episode"] = range(1, len(rewards) + 1)
    window = min(25, max(1, len(rewards) // 10))
    rewards["rolling_reward"] = rewards["r"].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rewards["episode"], rewards["r"], color="#9aa7b2", linewidth=0.8, alpha=0.45, label="Episode reward")
    ax.plot(rewards["episode"], rewards["rolling_reward"], color="#006d77", linewidth=2.0, label="Rolling mean")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined_learning_curves(log_root: str | Path, output_path: str | Path) -> None:
    log_root = Path(log_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for size_dir in sorted(log_root.glob("n_*")):
        rewards = _load_monitor_rewards(size_dir)
        if rewards.empty:
            continue
        rewards = rewards.sort_values("t").reset_index(drop=True)
        rewards["episode"] = range(1, len(rewards) + 1)
        window = min(25, max(1, len(rewards) // 10))
        rolling = rewards["r"].rolling(window=window, min_periods=1).mean()
        ax.plot(rewards["episode"], rolling, linewidth=2.0, label=size_dir.name)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title("PPO convergence by instance size")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling episode reward")
    ax.grid(alpha=0.25)
    ax.legend(title="Size")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def _load_monitor_rewards(log_dir: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in Path(log_dir).glob("*.monitor.csv"):
        try:
            frames.append(pd.read_csv(path, skiprows=1))
        except pd.errors.EmptyDataError:
            continue

    if not frames:
        return pd.DataFrame()

    rewards = pd.concat(frames, ignore_index=True)
    if not {"r", "l", "t"}.issubset(rewards.columns):
        return pd.DataFrame()
    return rewards
