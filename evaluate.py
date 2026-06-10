"""Evaluation utilities for PPO and heuristic policies."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baselines import PolicyResult, run_baselines
from instance_generator import ProblemInstance
from truck_drone_env import TruckDroneEnv
from vns import vns_policy


INSTALL_COMMAND = "/opt/anaconda3/bin/python -m pip install -r requirements.txt"


def load_ppo_model(model_path: str | Path) -> Any:
    maskable_error: Exception | None = None
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        maskable_error = exc
    else:
        try:
            return MaskablePPO.load(str(model_path))
        except Exception as exc:
            maskable_error = exc

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            "PPO evaluation requires sb3-contrib for MaskablePPO models or stable-baselines3 "
            f"for legacy PPO models. Missing import: {missing!r}. "
            f"Install project dependencies with: {INSTALL_COMMAND}"
        ) from exc

    try:
        return PPO.load(str(model_path))
    except Exception as exc:
        raise RuntimeError(
            "Unable to load model as MaskablePPO or legacy PPO. "
            f"MaskablePPO error: {maskable_error!r}; PPO error: {exc!r}"
        ) from exc


def evaluate_ppo_policy(
    model: Any,
    instance: ProblemInstance,
    size: int,
    max_nodes: int = 50,
    deterministic: bool = True,
    mask_actions: bool = True,
) -> PolicyResult:
    """Evaluate a saved Stable-Baselines3 PPO policy on one instance."""
    env = TruckDroneEnv(n_nodes=size, max_nodes=max_nodes)
    obs, _ = env.reset(options={"instance": instance})

    inference_start = time.perf_counter()
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    ppo_action_fallbacks = 0

    while not (terminated or truncated):
        action, used_fallback = _predict_ppo_action(
            model=model,
            obs=obs,
            action_mask=env.action_masks(),
            deterministic=deterministic,
            mask_actions=mask_actions,
        )
        ppo_action_fallbacks += int(used_fallback)
        obs, _, terminated, truncated, info = env.step(action)

    inference_runtime_seconds = time.perf_counter() - inference_start
    evaluation_start = time.perf_counter()
    service_sequence = [int(node) for node in info.get("service_sequence", env.service_sequence)]
    trace = env.get_trace()
    action_history = [int(action) for action in info.get("action_history", env.action_history)]
    truck_route = [int(node) for node in info.get("truck_route", env.truck_route)]
    drone_routes = [dict(route) for route in info.get("drone_routes", env.drone_routes)]
    route_signature = str(info.get("route_signature", env.route_signature))
    num_unique_actions = int(info.get("num_unique_actions", len(set(service_sequence))))
    evaluation_runtime_seconds = time.perf_counter() - evaluation_start
    runtime_seconds = inference_runtime_seconds + evaluation_runtime_seconds
    return PolicyResult(
        method="ppo",
        makespan=float(info.get("makespan", env.elapsed_time)),
        total_distance=float(info.get("total_distance", env.total_distance)),
        completion_rate=float(info.get("completion_rate", 0.0)),
        invalid_actions=int(info.get("invalid_actions", env.invalid_actions)),
        runtime=float(runtime_seconds),
        steps=int(info.get("step_count", env.step_count)),
        served_customers=int(info.get("served_customers", env.num_served)),
        unserved_customers=int(info.get("unserved_customers", env.num_unserved)),
        total_reward=float(info.get("total_reward", env.total_reward)),
        feasible=bool(info.get("feasible", False)),
        penalized_makespan=float(info.get("penalized_makespan", env.elapsed_time)),
        terminal_status=str(info.get("terminal_status", "unknown")),
        trace=trace,
        service_sequence=service_sequence,
        action_history=action_history,
        truck_route=truck_route,
        drone_routes=drone_routes,
        route_signature=route_signature,
        num_unique_actions=num_unique_actions,
        ppo_action_fallbacks=ppo_action_fallbacks,
        training_runtime_seconds=0.0,
        inference_runtime_seconds=float(inference_runtime_seconds),
        evaluation_runtime_seconds=float(evaluation_runtime_seconds),
    )


def evaluate_size(
    size: int,
    instances: list[ProblemInstance],
    model_path: str | Path | None = None,
    seed: int = 42,
    include_ortools: bool = True,
    include_vns: bool = True,
    vns_max_iterations: int = 200,
    vns_max_no_improve: int = 50,
    vns_time_limit_seconds: float | None = None,
    max_nodes: int = 50,
    mask_ppo_actions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Evaluate all requested methods for one instance size."""

    detailed_records: list[dict[str, object]] = []
    route_traces: dict[str, list[dict[str, Any]]] = {}
    ppo_available = model_path is not None and Path(model_path).exists()
    ppo_model = None
    ppo_training_runtime_seconds = 0.0
    if ppo_available:
        candidate_model = load_ppo_model(model_path)
        expected_env = TruckDroneEnv(n_nodes=size, max_nodes=max_nodes)
        if _model_matches_env(candidate_model, expected_env):
            ppo_model = candidate_model
            ppo_training_runtime_seconds = _load_training_runtime_seconds(model_path)
        else:
            print(
                f"Skipping PPO for n={size}: saved model shape does not match "
                "the current environment. Retrain this size first."
            )

    for instance_id, instance in enumerate(instances):
        baseline_results = run_baselines(
            instance,
            seed=seed + (10000 * size) + instance_id,
            include_ortools=include_ortools,
        )

        method_results: dict[str, PolicyResult] = dict(baseline_results)
        if include_vns:
            method_results["vns"] = vns_policy(
                instance=instance,
                seed=seed + (20_000 * size) + instance_id,
                max_nodes=max_nodes,
                max_iterations=vns_max_iterations,
                max_no_improve=vns_max_no_improve,
                time_limit_seconds=vns_time_limit_seconds,
            )
        if ppo_model is not None:
            method_results["ppo"] = evaluate_ppo_policy(
                model=ppo_model,
                instance=instance,
                size=size,
                max_nodes=max_nodes,
                mask_actions=mask_ppo_actions,
            )

        _warn_if_identical_instance_results(size, instance_id, method_results)

        for method, result in method_results.items():
            detailed_records.append(result.to_record(size, instance_id, instance.seed))
            if instance_id == 0:
                route_traces[method] = result.trace

    detailed = _add_nearest_neighbor_gap_columns(pd.DataFrame(detailed_records))
    summary = summarize_results(detailed)
    summary = _attach_ppo_training_runtime(summary, ppo_training_runtime_seconds)
    return summary, detailed, route_traces


def summarize_results(detailed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detailed records into paper-ready summary metrics."""

    if detailed.empty:
        return pd.DataFrame()

    detailed = _ensure_timing_columns(detailed)
    summary = (
        detailed.groupby(["size", "method"], as_index=False)
        .agg(
            mean_makespan=("makespan", "mean"),
            std_makespan=("makespan", "std"),
            mean_penalized_makespan=("penalized_makespan", "mean"),
            mean_total_distance=("total_distance", "mean"),
            completion_rate=("completion_rate", "mean"),
            feasibility_rate=("feasible", "mean"),
            mean_served_customers=("served_customers", "mean"),
            mean_unserved_customers=("unserved_customers", "mean"),
            mean_invalid_actions=("invalid_actions", "mean"),
            mean_total_reward=("total_reward", "mean"),
            mean_gap_to_nn_makespan_percent=("gap_to_nn_makespan_percent", "mean"),
            mean_gap_to_nn_distance_percent=("gap_to_nn_distance_percent", "mean"),
            mean_runtime=("runtime", "mean"),
            avg_runtime_seconds=("runtime_seconds", "mean"),
            std_runtime_seconds=("runtime_seconds", "std"),
            avg_inference_runtime_seconds=("inference_runtime_seconds", "mean"),
            avg_evaluation_runtime_seconds=("evaluation_runtime_seconds", "mean"),
            avg_training_runtime_seconds=("training_runtime_seconds", "mean"),
            avg_total_runtime_seconds=("total_runtime_seconds", "mean"),
        )
        .fillna({"std_makespan": 0.0, "std_runtime_seconds": 0.0})
    )
    summary = add_improvement_columns(summary)
    return summary.sort_values(
        ["size", "feasibility_rate", "completion_rate", "mean_penalized_makespan", "method"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)


def add_improvement_columns(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["improvement_vs_greedy"] = np.nan
    summary["improvement_vs_random"] = np.nan
    summary["penalized_improvement_vs_greedy"] = np.nan
    summary["penalized_improvement_vs_random"] = np.nan

    for size in summary["size"].unique():
        size_mask = summary["size"] == size
        size_rows = summary.loc[size_mask]
        greedy = _baseline_makespan(size_rows, "nearest_neighbor")
        random = _baseline_makespan(size_rows, "random_valid")
        greedy_penalized = _baseline_makespan(size_rows, "nearest_neighbor", "mean_penalized_makespan")
        random_penalized = _baseline_makespan(size_rows, "random_valid", "mean_penalized_makespan")

        feasible_mask = size_mask & (summary["feasibility_rate"] == 1.0)
        if greedy and greedy > 0:
            summary.loc[feasible_mask, "improvement_vs_greedy"] = (
                (greedy - summary.loc[feasible_mask, "mean_makespan"]) / greedy
            ) * 100.0
        if random and random > 0:
            summary.loc[feasible_mask, "improvement_vs_random"] = (
                (random - summary.loc[feasible_mask, "mean_makespan"]) / random
            ) * 100.0
        if greedy_penalized and greedy_penalized > 0:
            summary.loc[size_mask, "penalized_improvement_vs_greedy"] = (
                (greedy_penalized - summary.loc[size_mask, "mean_penalized_makespan"])
                / greedy_penalized
            ) * 100.0
        if random_penalized and random_penalized > 0:
            summary.loc[size_mask, "penalized_improvement_vs_random"] = (
                (random_penalized - summary.loc[size_mask, "mean_penalized_makespan"])
                / random_penalized
            ) * 100.0
    return summary


def _baseline_makespan(
    rows: pd.DataFrame,
    method: str,
    column: str = "mean_makespan",
) -> float | None:
    values = rows.loc[rows["method"] == method, column]
    if values.empty:
        return None
    return float(values.iloc[0])


def final_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "size",
        "method",
        "mean_makespan",
        "std_makespan",
        "mean_penalized_makespan",
        "mean_total_distance",
        "completion_rate",
        "feasibility_rate",
        "mean_served_customers",
        "mean_unserved_customers",
        "mean_invalid_actions",
        "mean_total_reward",
        "mean_gap_to_nn_makespan_percent",
        "mean_gap_to_nn_distance_percent",
        "mean_runtime",
        "avg_runtime_seconds",
        "std_runtime_seconds",
        "avg_inference_runtime_seconds",
        "avg_training_runtime_seconds",
        "improvement_vs_greedy",
    ]
    table = summary.loc[:, columns].copy()
    numeric_columns = [
        "mean_makespan",
        "std_makespan",
        "mean_penalized_makespan",
        "mean_total_distance",
        "completion_rate",
        "feasibility_rate",
        "mean_served_customers",
        "mean_unserved_customers",
        "mean_invalid_actions",
        "mean_total_reward",
        "mean_gap_to_nn_makespan_percent",
        "mean_gap_to_nn_distance_percent",
        "mean_runtime",
        "avg_runtime_seconds",
        "std_runtime_seconds",
        "avg_inference_runtime_seconds",
        "avg_training_runtime_seconds",
        "improvement_vs_greedy",
    ]
    table[numeric_columns] = table[numeric_columns].round(4)
    return table


def print_method_comparison(summary: pd.DataFrame) -> None:
    """Print compact method-level averages and PPO gaps."""

    if summary.empty:
        return

    method_averages = (
        summary.groupby("method", as_index=False)
        .agg(
            **{
                "Avg Makespan": ("mean_makespan", "mean"),
                "Avg Distance": ("mean_total_distance", "mean"),
                "Avg Runtime": ("avg_runtime_seconds", "mean"),
                "Feasible Rate": ("feasibility_rate", "mean"),
            }
        )
        .rename(columns={"method": "Method"})
        .sort_values("Avg Makespan")
    )
    display_columns = ["Avg Makespan", "Avg Distance", "Avg Runtime", "Feasible Rate"]
    method_averages[display_columns] = method_averages[display_columns].round(4)

    print("\nRuntime comparison table:")
    print(method_averages.to_string(index=False))

    gaps: list[dict[str, float | int]] = []
    for size, rows in summary.groupby("size"):
        method_to_makespan = dict(zip(rows["method"], rows["mean_makespan"]))
        ppo_makespan = method_to_makespan.get("ppo")
        if ppo_makespan is None:
            continue
        row: dict[str, float | int] = {"size": int(size)}
        for baseline in ("vns", "nearest_neighbor"):
            baseline_makespan = method_to_makespan.get(baseline)
            if baseline_makespan and baseline_makespan > 0:
                row[f"ppo_vs_{baseline}_gap_percent"] = (
                    100.0 * (ppo_makespan - baseline_makespan) / baseline_makespan
                )
        gaps.append(row)

    if gaps:
        gap_table = pd.DataFrame(gaps).round(4)
        print("\nPPO makespan gaps, positive means PPO is slower:")
        print(gap_table.to_string(index=False))


def _ensure_timing_columns(detailed: pd.DataFrame) -> pd.DataFrame:
    detailed = detailed.copy()
    if "runtime_seconds" not in detailed:
        detailed["runtime_seconds"] = detailed.get("runtime", 0.0)
    if "inference_runtime_seconds" not in detailed:
        detailed["inference_runtime_seconds"] = detailed["runtime_seconds"]
    if "evaluation_runtime_seconds" not in detailed:
        detailed["evaluation_runtime_seconds"] = 0.0
    if "training_runtime_seconds" not in detailed:
        detailed["training_runtime_seconds"] = 0.0
    if "total_runtime_seconds" not in detailed:
        detailed["total_runtime_seconds"] = (
            detailed["inference_runtime_seconds"]
            + detailed["evaluation_runtime_seconds"]
            + detailed["training_runtime_seconds"]
        )
    return detailed


def _load_training_runtime_seconds(model_path: str | Path | None) -> float:
    if model_path is None:
        return 0.0

    metadata_path = Path(model_path).with_suffix(".training.json")
    if not metadata_path.exists():
        return 0.0

    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0.0
    return float(metadata.get("training_runtime_seconds", 0.0))


def _attach_ppo_training_runtime(summary: pd.DataFrame, training_runtime_seconds: float) -> pd.DataFrame:
    if summary.empty:
        return summary

    summary = summary.copy()
    if "avg_training_runtime_seconds" not in summary:
        summary["avg_training_runtime_seconds"] = 0.0
    if training_runtime_seconds > 0:
        summary.loc[summary["method"] == "ppo", "avg_training_runtime_seconds"] = float(training_runtime_seconds)
    return summary


def _predict_ppo_action(
    model: Any,
    obs: np.ndarray,
    action_mask: np.ndarray,
    deterministic: bool,
    mask_actions: bool,
) -> tuple[int, bool]:
    if mask_actions:
        try:
            action, _ = model.predict(
                obs,
                action_masks=np.asarray(action_mask, dtype=bool),
                deterministic=deterministic,
            )
            return int(np.asarray(action).item()), False
        except TypeError:
            masked_action = _masked_discrete_action(
                model=model,
                obs=obs,
                action_mask=action_mask,
                deterministic=deterministic,
            )
            if masked_action is not None:
                return masked_action, True

    action, _ = model.predict(obs, deterministic=deterministic)
    return int(np.asarray(action).item()), bool(mask_actions)


def _masked_discrete_action(
    model: Any,
    obs: np.ndarray,
    action_mask: np.ndarray,
    deterministic: bool,
) -> int | None:
    valid_mask = np.asarray(action_mask, dtype=bool)
    if not valid_mask.any():
        return None

    try:
        import torch

        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
    except Exception:
        return None

    valid_mask = valid_mask[: len(probs)]
    if not valid_mask.any():
        return None

    masked_probs = np.where(valid_mask, probs[: len(valid_mask)], 0.0)
    if deterministic:
        return int(np.argmax(masked_probs))

    total_prob = masked_probs.sum()
    if total_prob <= 0:
        return int(np.flatnonzero(valid_mask)[0])
    masked_probs = masked_probs / total_prob
    return int(np.random.default_rng().choice(np.arange(len(masked_probs)), p=masked_probs))


def _model_matches_env(model: Any, env: TruckDroneEnv) -> bool:
    model_obs_shape = getattr(model.observation_space, "shape", None)
    env_obs_shape = getattr(env.observation_space, "shape", None)
    model_action_n = getattr(model.action_space, "n", None)
    env_action_n = getattr(env.action_space, "n", None)
    return model_obs_shape == env_obs_shape and model_action_n == env_action_n


def _add_nearest_neighbor_gap_columns(detailed: pd.DataFrame) -> pd.DataFrame:
    if detailed.empty:
        return detailed

    detailed = detailed.copy()
    nn_rows = detailed[detailed["method"] == "nearest_neighbor"][
        ["size", "instance_id", "makespan", "total_distance"]
    ].rename(
        columns={
            "makespan": "nn_makespan",
            "total_distance": "nn_distance",
        }
    )
    detailed = detailed.merge(nn_rows, on=["size", "instance_id"], how="left")
    detailed["gap_to_nn_makespan_percent"] = np.where(
        detailed["nn_makespan"] > 0,
        100.0 * (detailed["makespan"] - detailed["nn_makespan"]) / detailed["nn_makespan"],
        np.nan,
    )
    detailed["gap_to_nn_distance_percent"] = np.where(
        detailed["nn_distance"] > 0,
        100.0 * (detailed["total_distance"] - detailed["nn_distance"]) / detailed["nn_distance"],
        np.nan,
    )
    return detailed


def _warn_if_identical_instance_results(
    size: int,
    instance_id: int,
    method_results: dict[str, PolicyResult],
) -> None:
    if len(method_results) <= 1:
        return

    metric_pairs = {
        (
            round(result.makespan, 8),
            round(result.total_distance, 8),
        )
        for result in method_results.values()
    }
    if len(metric_pairs) == 1:
        methods = ", ".join(sorted(method_results))
        print(
            "WARNING: All methods have identical makespan and distance "
            f"for size={size}, instance={instance_id}: {methods}. "
            "Check whether route metrics depend on action sequence."
        )

    signatures = {result.route_signature for result in method_results.values()}
    if len(signatures) == 1:
        methods = ", ".join(sorted(method_results))
        print(
            "WARNING: All methods produced the same route signature "
            f"for size={size}, instance={instance_id}: {methods}."
        )
