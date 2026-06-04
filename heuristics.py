"""Compatibility exports for heuristic policies.

The project implementation lives in baselines.py; this module keeps the
heuristics.py name available for scripts or notes that refer to it.
"""

from baselines import (  # noqa: F401
    BASELINE_POLICIES,
    PolicyResult,
    drone_priority_policy,
    nearest_neighbor_policy,
    random_valid_policy,
    run_baselines,
)
