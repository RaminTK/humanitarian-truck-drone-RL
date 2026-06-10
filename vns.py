"""Variable Neighborhood Search baseline for truck-drone routing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from baselines import PolicyResult, nearest_neighbor_policy
from instance_generator import ProblemInstance
from truck_drone_env import TruckDroneEnv


Objective = tuple[float, float]
Neighborhood = Callable[[list[int], np.random.Generator], list[int]]


@dataclass(frozen=True)
class VNSConfig:
    """Search controls for the VNS metaheuristic."""

    max_iterations: int = 200
    max_no_improve: int = 50
    time_limit_seconds: float | None = None
    local_search_passes: int = 2
    local_search_trials_per_neighborhood: int = 2


def vns_policy(
    instance: ProblemInstance,
    seed: int | None = None,
    max_nodes: int | None = None,
    max_iterations: int = 200,
    max_no_improve: int = 50,
    time_limit_seconds: float | None = None,
) -> PolicyResult:
    """Optimize a route order with VNS and evaluate it in TruckDroneEnv."""

    config = VNSConfig(
        max_iterations=max_iterations,
        max_no_improve=max_no_improve,
        time_limit_seconds=time_limit_seconds,
    )
    max_nodes = int(max_nodes or instance.n_nodes)
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()
    objective_cache: dict[tuple[int, ...], Objective] = {}

    initial_solution = nearest_neighbor_policy(instance).service_sequence
    if not is_feasible_sequence(initial_solution, instance.n_nodes):
        initial_solution = list(range(instance.n_nodes))

    best_sequence = list(initial_solution)
    best_objective = _objective_for_sequence(
        instance=instance,
        sequence=best_sequence,
        max_nodes=max_nodes,
        objective_cache=objective_cache,
    )

    iteration = 0
    no_improve = 0
    neighborhoods = _neighborhoods()

    while iteration < config.max_iterations and no_improve < config.max_no_improve:
        if _time_limit_reached(start_time, config.time_limit_seconds):
            break

        k = 0
        improved = False
        while k < len(neighborhoods):
            if _time_limit_reached(start_time, config.time_limit_seconds):
                break

            candidate = neighborhoods[k](best_sequence, rng)
            if not is_feasible_sequence(candidate, instance.n_nodes):
                k += 1
                continue

            candidate, candidate_objective = _local_search(
                instance=instance,
                sequence=candidate,
                rng=rng,
                max_nodes=max_nodes,
                config=config,
                objective_cache=objective_cache,
                start_time=start_time,
            )
            if _improves(candidate_objective, best_objective):
                best_sequence = candidate
                best_objective = candidate_objective
                k = 0
                improved = True
            else:
                k += 1

        no_improve = 0 if improved else no_improve + 1
        iteration += 1

    inference_runtime_seconds = time.perf_counter() - start_time
    return sequence_to_policy_result(
        instance=instance,
        sequence=best_sequence,
        max_nodes=max_nodes,
        runtime=inference_runtime_seconds,
        method="vns",
        inference_runtime_seconds=inference_runtime_seconds,
    )


def evaluate_sequence_in_env(
    instance: ProblemInstance,
    sequence: list[int],
    max_nodes: int | None = None,
) -> dict[str, Any]:
    """Execute a candidate service sequence in the project environment."""

    max_nodes = int(max_nodes or instance.n_nodes)
    env = TruckDroneEnv(n_nodes=instance.n_nodes, max_nodes=max_nodes)
    obs, info = env.reset(options={"instance": instance})
    terminated = False
    truncated = False

    for node in sequence:
        if terminated or truncated:
            break
        obs, _, terminated, truncated, info = env.step(int(node))

    if not (terminated or truncated):
        terminal_action = env.terminal_action
        if env.action_masks()[terminal_action]:
            obs, _, terminated, truncated, info = env.step(terminal_action)

    return dict(info)


def sequence_to_policy_result(
    *,
    instance: ProblemInstance,
    sequence: list[int],
    max_nodes: int,
    runtime: float,
    method: str,
    inference_runtime_seconds: float | None = None,
    evaluation_runtime_seconds: float | None = None,
) -> PolicyResult:
    evaluation_start = time.perf_counter()
    info = evaluate_sequence_in_env(instance=instance, sequence=sequence, max_nodes=max_nodes)
    measured_evaluation_runtime_seconds = time.perf_counter() - evaluation_start
    evaluation_runtime_seconds = (
        measured_evaluation_runtime_seconds
        if evaluation_runtime_seconds is None
        else evaluation_runtime_seconds
    )
    inference_runtime_seconds = float(runtime if inference_runtime_seconds is None else inference_runtime_seconds)
    total_runtime_seconds = inference_runtime_seconds + float(evaluation_runtime_seconds)
    service_sequence = [int(node) for node in info.get("service_sequence", sequence)]
    return PolicyResult(
        method=method,
        makespan=float(info.get("makespan", 0.0)),
        total_distance=float(info.get("total_distance", 0.0)),
        completion_rate=float(info.get("completion_rate", 0.0)),
        invalid_actions=int(info.get("invalid_actions", 0)),
        runtime=float(total_runtime_seconds),
        steps=int(info.get("step_count", len(sequence))),
        served_customers=int(info.get("served_customers", len(service_sequence))),
        unserved_customers=int(info.get("unserved_customers", 0)),
        total_reward=float(info.get("total_reward", 0.0)),
        feasible=bool(info.get("feasible", False)),
        penalized_makespan=float(info.get("penalized_makespan", info.get("makespan", 0.0))),
        terminal_status=str(info.get("terminal_status", "unknown")),
        trace=[dict(step) for step in info.get("route_history", [])],
        service_sequence=service_sequence,
        action_history=[int(action) for action in info.get("action_history", sequence)],
        truck_route=[int(node) for node in info.get("truck_route", [])],
        drone_routes=[dict(route) for route in info.get("drone_routes", [])],
        route_signature=str(info.get("route_signature", "-".join(str(node) for node in service_sequence))),
        num_unique_actions=int(info.get("num_unique_actions", len(set(service_sequence)))),
        training_runtime_seconds=0.0,
        inference_runtime_seconds=inference_runtime_seconds,
        evaluation_runtime_seconds=float(evaluation_runtime_seconds),
    )


def swap_neighbor(sequence: list[int], rng: np.random.Generator) -> list[int]:
    candidate = list(sequence)
    if len(candidate) < 2:
        return candidate
    first, second = rng.choice(len(candidate), size=2, replace=False)
    candidate[int(first)], candidate[int(second)] = candidate[int(second)], candidate[int(first)]
    return candidate


def relocate_neighbor(sequence: list[int], rng: np.random.Generator) -> list[int]:
    candidate = list(sequence)
    if len(candidate) < 2:
        return candidate
    source, target = rng.choice(len(candidate), size=2, replace=False)
    node = candidate.pop(int(source))
    candidate.insert(int(target), node)
    return candidate


def two_opt_neighbor(sequence: list[int], rng: np.random.Generator) -> list[int]:
    candidate = list(sequence)
    if len(candidate) < 3:
        return candidate
    first, second = sorted(rng.choice(len(candidate), size=2, replace=False))
    first = int(first)
    second = int(second)
    if first == second:
        return candidate
    candidate[first : second + 1] = reversed(candidate[first : second + 1])
    return candidate


def is_feasible_sequence(sequence: list[int], n_nodes: int) -> bool:
    if len(sequence) != n_nodes:
        return False
    return set(sequence) == set(range(n_nodes))


def _local_search(
    *,
    instance: ProblemInstance,
    sequence: list[int],
    rng: np.random.Generator,
    max_nodes: int,
    config: VNSConfig,
    objective_cache: dict[tuple[int, ...], Objective],
    start_time: float,
) -> tuple[list[int], Objective]:
    current = list(sequence)
    current_objective = _objective_for_sequence(
        instance=instance,
        sequence=current,
        max_nodes=max_nodes,
        objective_cache=objective_cache,
    )
    neighborhoods = _neighborhoods()

    for _ in range(config.local_search_passes):
        if _time_limit_reached(start_time, config.time_limit_seconds):
            break

        improved = False
        for neighborhood_index in rng.permutation(len(neighborhoods)):
            neighborhood = neighborhoods[int(neighborhood_index)]
            for _ in range(config.local_search_trials_per_neighborhood):
                candidate = neighborhood(current, rng)
                if not is_feasible_sequence(candidate, instance.n_nodes):
                    continue
                candidate_objective = _objective_for_sequence(
                    instance=instance,
                    sequence=candidate,
                    max_nodes=max_nodes,
                    objective_cache=objective_cache,
                )
                if _improves(candidate_objective, current_objective):
                    current = candidate
                    current_objective = candidate_objective
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return current, current_objective


def _objective_for_sequence(
    *,
    instance: ProblemInstance,
    sequence: list[int],
    max_nodes: int,
    objective_cache: dict[tuple[int, ...], Objective],
) -> Objective:
    key = tuple(int(node) for node in sequence)
    if key in objective_cache:
        return objective_cache[key]

    if not is_feasible_sequence(list(key), instance.n_nodes):
        objective_cache[key] = (float("inf"), float("inf"))
        return objective_cache[key]

    info = evaluate_sequence_in_env(instance=instance, sequence=list(key), max_nodes=max_nodes)
    if (
        info.get("completion_rate", 0.0) < 1.0
        or info.get("invalid_actions", 0) != 0
        or not info.get("feasible", False)
    ):
        objective_cache[key] = (float("inf"), float("inf"))
        return objective_cache[key]

    objective_cache[key] = (
        float(info.get("makespan", float("inf"))),
        float(info.get("total_distance", float("inf"))),
    )
    return objective_cache[key]


def _neighborhoods() -> list[Neighborhood]:
    return [swap_neighbor, relocate_neighbor, two_opt_neighbor]


def _improves(candidate: Objective, incumbent: Objective) -> bool:
    return candidate < incumbent


def _time_limit_reached(start_time: float, time_limit_seconds: float | None) -> bool:
    if time_limit_seconds is None:
        return False
    return (time.perf_counter() - start_time) >= time_limit_seconds
