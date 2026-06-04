"""Synthetic instance generation for humanitarian truck-drone routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEPOT = np.array([50.0, 50.0], dtype=np.float32)
DEFAULT_TRUCK_SPEED = 1.0
DEFAULT_DRONE_SPEED = 2.0
DEFAULT_TRUCK_ACCESSIBLE_PROB = 0.65
INSTANCE_SIZES = (10, 20, 30, 40, 50)


@dataclass(frozen=True)
class ProblemInstance:
    """One synthetic truck-drone routing instance."""

    coords: np.ndarray
    truck_accessible: np.ndarray
    demand: np.ndarray
    svi: np.ndarray
    depot: np.ndarray = field(default_factory=lambda: DEPOT.copy())
    truck_speed: float = DEFAULT_TRUCK_SPEED
    drone_speed: float = DEFAULT_DRONE_SPEED
    seed: int | None = None

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=np.float32)
        truck_accessible = np.asarray(self.truck_accessible, dtype=bool)
        demand = np.asarray(self.demand, dtype=np.float32)
        svi = np.asarray(self.svi, dtype=np.float32)
        depot = np.asarray(self.depot, dtype=np.float32)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (n_nodes, 2)")
        if len(truck_accessible) != len(coords):
            raise ValueError("truck_accessible length must match coords")
        if len(demand) != len(coords):
            raise ValueError("demand length must match coords")
        if len(svi) != len(coords):
            raise ValueError("svi length must match coords")
        if depot.shape != (2,):
            raise ValueError("depot must have shape (2,)")
        if self.truck_speed <= 0 or self.drone_speed <= 0:
            raise ValueError("vehicle speeds must be positive")

        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "truck_accessible", truck_accessible)
        object.__setattr__(self, "demand", demand)
        object.__setattr__(self, "svi", svi)
        object.__setattr__(self, "depot", depot)

    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def drone_only(self) -> np.ndarray:
        return ~self.truck_accessible

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "node": np.arange(self.n_nodes),
                "x": self.coords[:, 0],
                "y": self.coords[:, 1],
                "truck_accessible": self.truck_accessible,
                "drone_only": self.drone_only,
                "demand": self.demand,
                "svi": self.svi,
                "seed": self.seed,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "coords": self.coords.tolist(),
            "truck_accessible": self.truck_accessible.tolist(),
            "demand": self.demand.tolist(),
            "svi": self.svi.tolist(),
            "depot": self.depot.tolist(),
            "truck_speed": self.truck_speed,
            "drone_speed": self.drone_speed,
            "seed": self.seed,
        }


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).sum())


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def truck_travel_time(instance: ProblemInstance, origin: np.ndarray, node_index: int) -> float:
    distance = manhattan_distance(origin, instance.coords[node_index])
    return distance / instance.truck_speed


def drone_round_trip_time(instance: ProblemInstance, origin: np.ndarray, node_index: int) -> float:
    distance = 2.0 * euclidean_distance(origin, instance.coords[node_index])
    return distance / instance.drone_speed


def generate_instance(
    n_nodes: int,
    seed: int | None = None,
    truck_accessible_prob: float = DEFAULT_TRUCK_ACCESSIBLE_PROB,
    depot: np.ndarray = DEPOT,
    truck_speed: float = DEFAULT_TRUCK_SPEED,
    drone_speed: float = DEFAULT_DRONE_SPEED,
) -> ProblemInstance:
    """Generate one uniformly random instance with reproducible seed control."""

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if not 0.0 <= truck_accessible_prob <= 1.0:
        raise ValueError("truck_accessible_prob must be in [0, 1]")

    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, size=(n_nodes, 2)).astype(np.float32)
    truck_accessible = rng.random(n_nodes) < truck_accessible_prob
    demand = np.ones(n_nodes, dtype=np.float32)
    svi = rng.uniform(0.0, 1.0, size=n_nodes).astype(np.float32)

    return ProblemInstance(
        coords=coords,
        truck_accessible=truck_accessible,
        demand=demand,
        svi=svi,
        depot=np.asarray(depot, dtype=np.float32),
        truck_speed=truck_speed,
        drone_speed=drone_speed,
        seed=seed,
    )


def generate_instances(
    n_nodes: int,
    count: int,
    seed: int | None = None,
    truck_accessible_prob: float = DEFAULT_TRUCK_ACCESSIBLE_PROB,
) -> list[ProblemInstance]:
    """Generate a list of independent instances."""

    if count <= 0:
        raise ValueError("count must be positive")

    rng = np.random.default_rng(seed)
    child_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=count, dtype=np.uint32)
    return [
        generate_instance(
            n_nodes=n_nodes,
            seed=int(child_seed),
            truck_accessible_prob=truck_accessible_prob,
        )
        for child_seed in child_seeds
    ]


def save_instances(instances: Iterable[ProblemInstance], output_dir: str | Path) -> None:
    """Save generated instances as one CSV per instance for auditability."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for index, instance in enumerate(instances):
        instance.to_frame().to_csv(path / f"instance_{index:04d}.csv", index=False)
