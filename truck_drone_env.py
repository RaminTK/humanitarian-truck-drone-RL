"""Gymnasium environment for the single-truck single-drone routing study.

Model limitations in this first version:
- one truck and one drone
- drone serves one node per sortie
- truck waits during drone service
- no battery, charging, payload, or multi-agent coordination constraints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from instance_generator import (
    DEPOT,
    ProblemInstance,
    drone_round_trip_time,
    euclidean_distance,
    generate_instance,
    manhattan_distance,
    truck_travel_time,
)


@dataclass(frozen=True)
class RewardConfig:
    """Reward terms ordered around feasibility first and route cost second."""

    service_reward: float = 100.0
    completion_bonus: float = 1000.0
    invalid_action_penalty: float = 500.0
    unserved_penalty: float = 5000.0
    time_weight: float = 0.5
    distance_weight: float = 1.0
    nearest_neighbor_regret_weight: float = 2.0
    nearest_neighbor_bonus: float = 25.0
    final_makespan_weight: float = 0.1
    final_distance_weight: float = 0.1
    step_penalty: float = 1.0


DEFAULT_REWARD_CONFIG = RewardConfig()
PENALIZED_MAKESPAN_UNSERVED_WEIGHT = 10_000.0
PENALIZED_MAKESPAN_INVALID_WEIGHT = 1_000.0
DEPOT_NODE_ID = -1


class TruckDroneEnv(gym.Env):
    """A fixed-observation Gymnasium environment for truck-drone routing."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_nodes: int = 10,
        max_nodes: int = 50,
        instance_pool: list[ProblemInstance] | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
        time_normalizer: float = 500.0,
        invalid_time_cost: float = 1.0,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
    ) -> None:
        super().__init__()
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        if n_nodes > max_nodes:
            raise ValueError("n_nodes cannot exceed max_nodes")

        self.n_nodes = n_nodes
        self.max_nodes = max_nodes
        self.instance_pool = instance_pool
        self.max_steps = max_steps or 3 * n_nodes
        self.time_normalizer = time_normalizer
        self.invalid_time_cost = invalid_time_cost
        self.reward_config = reward_config
        self.rng = np.random.default_rng(seed)

        self.terminal_action = max_nodes
        self.action_space = spaces.Discrete(max_nodes + 1)
        obs_dim = 2 + max_nodes + max_nodes + (2 * max_nodes) + (3 * max_nodes) + (max_nodes + 1) + 5
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.full(obs_dim, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.instance: ProblemInstance | None = None
        self.current_position = DEPOT.copy()
        self.visited = np.zeros(self.n_nodes, dtype=bool)
        self.elapsed_time = 0.0
        self.total_distance = 0.0
        self.total_reward = 0.0
        self.invalid_actions = 0
        self.step_count = 0
        self.current_node = DEPOT_NODE_ID
        self.route_trace: list[dict[str, Any]] = []
        self.action_history: list[int] = []
        self.truck_route: list[int] = [DEPOT_NODE_ID]
        self.drone_routes: list[dict[str, Any]] = []
        self.service_sequence: list[int] = []
        self.travel_history: list[dict[str, Any]] = []
        self.total_regret = 0.0
        self.last_regret = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        self.instance = self._select_instance(options)
        if self.instance.n_nodes != self.n_nodes:
            raise ValueError("instance node count must match environment n_nodes")

        self.current_position = self.instance.depot.copy()
        self.visited = np.zeros(self.n_nodes, dtype=bool)
        self.elapsed_time = 0.0
        self.total_distance = 0.0
        self.total_reward = 0.0
        self.invalid_actions = 0
        self.step_count = 0
        self.current_node = DEPOT_NODE_ID
        self.route_trace = []
        self.action_history = []
        self.truck_route = [DEPOT_NODE_ID]
        self.drone_routes = []
        self.service_sequence = []
        self.travel_history = []
        self.total_regret = 0.0
        self.last_regret = 0.0

        return self._get_obs(), self._get_info(valid_action=True, service_mode="reset")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.instance is None:
            raise RuntimeError("reset must be called before step")

        action = int(action)
        self.step_count += 1
        reward = -self.reward_config.step_penalty
        terminated = False
        truncated = False
        valid_action = False
        service_mode = "invalid"
        travel_time = 0.0
        distance = 0.0
        served_new_customer = False
        terminal_status = "running"
        attempted_early_termination = False

        self.last_regret = 0.0

        if action == self.terminal_action:
            if self.visited.all():
                valid_action = True
                service_mode = "noop_complete"
                terminated = True
                terminal_status = "completed"
            else:
                attempted_early_termination = True
                self._apply_invalid_action()
        elif 0 <= action < self.n_nodes and not self.visited[action]:
            valid_action = True
            served_new_customer = True
            nearest_action, nearest_distance = self._nearest_feasible_action()
            service_mode, travel_time, distance = self._serve_node(action=action, node_index=action)
            regret = max(0.0, distance - nearest_distance)
            self.last_regret = regret
            self.total_regret += regret
            reward += self.reward_config.service_reward
            reward -= self.reward_config.time_weight * travel_time
            reward -= self.reward_config.distance_weight * distance
            reward -= self.reward_config.nearest_neighbor_regret_weight * regret
            if nearest_action is not None and action == nearest_action:
                reward += self.reward_config.nearest_neighbor_bonus
        else:
            self._apply_invalid_action()

        all_served = self.visited.all()
        if all_served and not terminated:
            terminated = True
            terminal_status = "completed"

        if attempted_early_termination and not all_served:
            truncated = True
            terminal_status = "early_termination"
        elif self.step_count >= self.max_steps and not terminated:
            truncated = True
            terminal_status = "max_steps"

        if terminated and all_served:
            reward += self.reward_config.completion_bonus
            reward -= self.reward_config.final_makespan_weight * self.elapsed_time
            reward -= self.reward_config.final_distance_weight * self.total_distance

        if truncated and not all_served:
            reward -= self.reward_config.unserved_penalty * self.num_unserved

        if not valid_action:
            reward -= self.reward_config.invalid_action_penalty
            reward -= self.reward_config.time_weight * self.invalid_time_cost

        self.total_reward += float(reward)

        info = self._get_info(
            valid_action=valid_action,
            service_mode=service_mode,
            travel_time=travel_time,
            distance=distance,
            served_new_customer=served_new_customer,
            terminal_status=terminal_status,
            step_reward=float(reward),
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    def _select_instance(self, options: dict[str, Any]) -> ProblemInstance:
        if "instance" in options:
            return options["instance"]

        if self.instance_pool:
            if "instance_index" in options:
                return self.instance_pool[int(options["instance_index"])]
            index = int(self.rng.integers(0, len(self.instance_pool)))
            return self.instance_pool[index]

        seed = int(self.rng.integers(0, np.iinfo(np.uint32).max))
        return generate_instance(self.n_nodes, seed=seed)

    def _get_obs(self) -> np.ndarray:
        if self.instance is None:
            raise RuntimeError("instance is not initialized")

        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        cursor = 0

        obs[cursor : cursor + 2] = np.clip(self.current_position / 100.0, 0.0, 1.0)
        cursor += 2

        obs[cursor : cursor + self.n_nodes] = self.visited.astype(np.float32)
        cursor += self.max_nodes

        obs[cursor : cursor + self.n_nodes] = self.instance.truck_accessible.astype(np.float32)
        cursor += self.max_nodes

        coords = np.zeros((self.max_nodes, 2), dtype=np.float32)
        coords[: self.n_nodes] = self.instance.coords / 100.0
        obs[cursor : cursor + (2 * self.max_nodes)] = coords.reshape(-1)
        cursor += 2 * self.max_nodes

        service_distances = np.zeros(self.max_nodes, dtype=np.float32)
        service_times = np.zeros(self.max_nodes, dtype=np.float32)
        nearest_indicator = np.zeros(self.max_nodes, dtype=np.float32)
        nearest_action, _ = self._nearest_feasible_action()
        for node_index in range(self.n_nodes):
            if self.visited[node_index]:
                continue
            _, service_time, service_distance = self._choose_service_mode(node_index)
            service_distances[node_index] = min(service_distance / 250.0, 10.0)
            service_times[node_index] = min(service_time / self.time_normalizer, 10.0)
        if nearest_action is not None:
            nearest_indicator[nearest_action] = 1.0

        obs[cursor : cursor + self.max_nodes] = service_distances
        cursor += self.max_nodes
        obs[cursor : cursor + self.max_nodes] = service_times
        cursor += self.max_nodes
        obs[cursor : cursor + self.max_nodes] = nearest_indicator
        cursor += self.max_nodes

        mask = np.zeros(self.max_nodes + 1, dtype=np.float32)
        action_mask = self.get_action_mask().astype(np.float32)
        mask[:] = action_mask
        obs[cursor : cursor + self.max_nodes + 1] = mask
        cursor += self.max_nodes + 1

        remaining_ratio = self.num_unserved / self.n_nodes
        served_ratio = self.num_served / self.n_nodes
        elapsed_normalized = min(self.elapsed_time / self.time_normalizer, 10.0)
        obs[cursor] = remaining_ratio
        obs[cursor + 1] = elapsed_normalized
        obs[cursor + 2] = served_ratio
        obs[cursor + 3] = 1.0
        obs[cursor + 4] = min(self.invalid_actions / max(1, self.max_steps), 10.0)
        return obs

    @property
    def num_served(self) -> int:
        return int(self.visited.sum())

    @property
    def num_unserved(self) -> int:
        return int(self.n_nodes - self.num_served)

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[: self.n_nodes] = ~self.visited
        if self.visited.all():
            mask[self.terminal_action] = True
        if not mask.any():
            fallback_action = self.terminal_action if self.visited.all() else int(np.flatnonzero(~self.visited)[0])
            mask[fallback_action] = True
        return mask

    def action_masks(self) -> np.ndarray:
        return self.get_action_mask().astype(bool, copy=False)

    def _serve_node(self, action: int, node_index: int) -> tuple[str, float, float]:
        if self.instance is None:
            raise RuntimeError("instance is not initialized")

        origin = self.current_position.copy()
        from_node = self.current_node
        node_position = self.instance.coords[node_index]
        service_mode, travel_time, distance = self._choose_service_mode(node_index)

        if service_mode == "truck":
            self.current_position = node_position.copy()
            self.current_node = node_index
            self.truck_route.append(node_index)
        else:
            self.drone_routes.append(
                {
                    "step": int(self.step_count),
                    "launch_node": int(from_node),
                    "served_node": int(node_index),
                    "return_node": int(from_node),
                    "launch_x": float(origin[0]),
                    "launch_y": float(origin[1]),
                    "served_x": float(node_position[0]),
                    "served_y": float(node_position[1]),
                    "distance": float(distance),
                    "time": float(travel_time),
                }
            )

        self.elapsed_time += travel_time
        self.total_distance += distance
        self.visited[node_index] = True
        self.action_history.append(int(action))
        self.service_sequence.append(int(node_index))
        travel_record = {
            "step": int(self.step_count),
            "action": int(action),
            "served_node": int(node_index),
            "node": int(node_index),
            "vehicle": service_mode,
            "mode": service_mode,
            "from_node": int(from_node),
            "to_node": int(node_index),
            "from_x": float(origin[0]),
            "from_y": float(origin[1]),
            "to_x": float(node_position[0]),
            "to_y": float(node_position[1]),
            "incremental_distance": float(distance),
            "incremental_time": float(travel_time),
            "distance": float(distance),
            "service_time": float(travel_time),
            "cumulative_distance": float(self.total_distance),
            "current_time": float(self.elapsed_time),
            "elapsed_after": float(self.elapsed_time),
        }
        self.route_trace.append(travel_record)
        self.travel_history.append(travel_record)
        return service_mode, travel_time, distance

    def _choose_service_mode(self, node_index: int) -> tuple[str, float, float]:
        if self.instance is None:
            raise RuntimeError("instance is not initialized")

        if self.instance.truck_accessible[node_index]:
            distance = manhattan_distance(self.current_position, self.instance.coords[node_index])
            return "truck", distance / self.instance.truck_speed, distance

        distance = 2.0 * euclidean_distance(self.current_position, self.instance.coords[node_index])
        return "drone", distance / self.instance.drone_speed, distance

    def _service_distance(self, node_index: int) -> float:
        if self.instance is None:
            raise RuntimeError("instance is not initialized")
        if self.instance.truck_accessible[node_index]:
            return manhattan_distance(self.current_position, self.instance.coords[node_index])
        return 2.0 * euclidean_distance(self.current_position, self.instance.coords[node_index])

    def _nearest_feasible_action(self) -> tuple[int | None, float]:
        candidates = np.flatnonzero(~self.visited)
        if len(candidates) == 0:
            return None, 0.0

        best_action = int(candidates[0])
        best_distance = self._service_distance(best_action)
        for candidate in candidates[1:]:
            candidate = int(candidate)
            candidate_distance = self._service_distance(candidate)
            if (candidate_distance, candidate) < (best_distance, best_action):
                best_action = candidate
                best_distance = candidate_distance
        return best_action, best_distance

    def _apply_invalid_action(self) -> None:
        self.invalid_actions += 1
        self.elapsed_time += self.invalid_time_cost

    def _get_info(
        self,
        *,
        valid_action: bool,
        service_mode: str,
        travel_time: float = 0.0,
        distance: float = 0.0,
        served_new_customer: bool = False,
        terminal_status: str = "running",
        step_reward: float = 0.0,
    ) -> dict[str, Any]:
        completion_rate = self.num_served / self.n_nodes
        feasible = completion_rate == 1.0 and self.invalid_actions == 0
        penalized_makespan = (
            self.elapsed_time
            + (PENALIZED_MAKESPAN_UNSERVED_WEIGHT * self.num_unserved)
            + (PENALIZED_MAKESPAN_INVALID_WEIGHT * self.invalid_actions)
        )
        return {
            "valid_action": valid_action,
            "service_mode": service_mode,
            "travel_time": float(travel_time),
            "distance": float(distance),
            "served_new_customer": bool(served_new_customer),
            "elapsed_time": float(self.elapsed_time),
            "makespan": float(self.elapsed_time),
            "total_distance": float(self.total_distance),
            "total_reward": float(self.total_reward),
            "total_regret": float(self.total_regret),
            "average_regret": float(self.total_regret / max(1, self.num_served)),
            "nearest_neighbor_regret": float(self.last_regret),
            "step_reward": float(step_reward),
            "invalid_actions": int(self.invalid_actions),
            "all_served": bool(self.visited.all()),
            "served_customers": int(self.num_served),
            "unserved_customers": int(self.num_unserved),
            "completion_rate": float(completion_rate),
            "feasible": bool(feasible),
            "penalized_makespan": float(penalized_makespan),
            "remaining_nodes": int((~self.visited).sum()),
            "action_mask": self.get_action_mask().astype(np.int8),
            "step_count": int(self.step_count),
            "terminal_status": terminal_status,
            "route_history": self.get_trace(),
            "action_history": list(self.action_history),
            "service_sequence": list(self.service_sequence),
            "truck_route": list(self.truck_route),
            "drone_routes": [dict(route) for route in self.drone_routes],
            "travel_history": [dict(step) for step in self.travel_history],
            "route_signature": self.route_signature,
            "num_unique_actions": int(len(set(self.action_history))),
        }

    def get_trace(self) -> list[dict[str, Any]]:
        return [dict(step) for step in self.route_trace]

    @property
    def route_signature(self) -> str:
        return "-".join(str(node) for node in self.service_sequence)

    def render(self) -> str:
        served = int(self.visited.sum())
        return (
            f"served={served}/{self.n_nodes}, "
            f"elapsed_time={self.elapsed_time:.2f}, "
            f"invalid_actions={self.invalid_actions}"
        )
