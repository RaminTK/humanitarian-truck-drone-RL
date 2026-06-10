# Deep Reinforcement Learning for Dynamic Humanitarian Truck-Drone Routing under Infrastructure Disruptions

This project is a complete Python 3.12 research scaffold for a single-objective deep reinforcement learning study of humanitarian truck-drone routing. It generates synthetic disrupted-access delivery instances, trains PPO policies, compares them against heuristic baselines, and saves conference-paper-ready result tables and plots.

## Version Safety Note

- `v1` is the stable baseline checkpoint: MaskablePPO, route-based evaluation, VNS, and detailed runtime reporting.
- `v2-ppo-improvement` is the experimental branch for PPO improvement and ablation studies.
- Experimental v2 work should write to `results_v2/` and should not overwrite v1 `results/`, `models/`, or baseline source files unless the change is explicitly part of the v2 experiment.

## Problem Description

The setting contains one depot, one truck, one drone, and `n` aid nodes. Some nodes are truck-accessible and the rest are drone-only. Every node must be served exactly once.

- Depot: `(50, 50)`
- Aid nodes: uniformly generated in `[0, 100] x [0, 100]`
- Truck accessibility probability: `0.65`
- Demand: `1` for every node
- SVI score: sampled in `[0, 1]`, recorded but not used in the objective
- Truck speed: `1.0`
- Drone speed: `2.0`
- Truck distance: Manhattan distance
- Drone distance: Euclidean distance
- Objective: minimize total mission completion time, equivalent to makespan

The truck can serve only truck-accessible nodes. The drone can serve any node. For each drone service, the drone launches from the current truck location, serves one node, and returns to the same truck location while the truck waits.

## Environment Design

`truck_drone_env.py` implements `TruckDroneEnv` using `gymnasium.Env`.

The action space is fixed as `Discrete(max_nodes + 1)`:

- `0` to `n - 1`: choose the next node to serve
- `n` to `max_nodes - 1`: padded invalid actions for smaller instances
- `max_nodes`: no-op, valid only when all nodes are already served

For truck-accessible selected nodes, the environment uses a deterministic route-sensitive service rule:

- Truck-accessible nodes are served by truck, so the truck position changes and route order affects future costs
- Drone-only nodes are served by drone from the current truck location, and the truck waits for the sortie to return

This keeps the action space simple while ensuring makespan and distance are computed from the realized ordered route rather than only from the set of served nodes.

The observation is a fixed-size vector padded to `max_nodes=50`:

- Current truck `x` and `y`, normalized by `100`
- Visited indicator for each node
- Truck-accessible indicator for each node
- Flattened node coordinates, normalized by `100`
- Per-node feasible service distance from the current truck location
- Per-node feasible service time from the current truck location
- One-hot nearest-feasible-node indicator for dense routing guidance
- Valid-action mask over the fixed `max_nodes + 1` action space, with padding actions disabled and the terminal/no-op action enabled only after all nodes are served
- Remaining node ratio
- Current elapsed time, normalized
- Served node ratio
- Drone availability, always `1.0` because the drone returns after each sortie
- Invalid-action ratio

This lets MaskablePPO use one fixed action space across instance sizes for curriculum training.

## Reward Function

The dense reward prioritizes feasibility first and route quality second:

- Valid new-customer service reward: `+100`
- Travel time penalty: `-0.5 * incremental_time`
- Distance penalty: `-1.0 * incremental_distance`
- Nearest-neighbor regret penalty: `-2.0 * max(0, chosen_distance - nearest_feasible_distance)`
- Nearest-neighbor imitation bonus: `+25` when the selected action matches the nearest feasible customer
- Invalid action penalty: `-500`
- Step penalty: `-1`
- Completion bonus: `+1000`
- Final route-quality penalty after completion: `-0.1 * makespan - 0.1 * total_distance`
- Failed terminal penalty: `-5000 * unserved_customers`

Episodes terminate successfully only when all nodes are served. If `max_steps = 3n` is reached first, the episode is truncated and receives the unserved-customer penalty. The terminal/no-op action is valid only after all nodes are served; choosing it early is treated as a failed early termination.

## Baselines

`baselines.py` implements:

- `random_valid`: uniformly chooses among unvisited nodes
- `nearest_neighbor`: chooses the unvisited node with minimum feasible service time
- `drone_priority`: prioritizes drone-only and drone-favorable nodes, then nearby truck nodes
- `vns`: Variable Neighborhood Search initialized from the nearest-neighbor route
- `ortools_tsp`: optional OR-Tools route over truck-accessible nodes plus drone-only sorties from stops

If OR-Tools is not installed, the optional baseline is skipped automatically.

## Outputs

Experiment outputs are saved under `results/`:

- `results/summary.csv`
- `results/detailed_results.csv`
- `results/plots/learning_curve_n<size>.png`
- `results/plots/convergence_reward_curve.png`
- `results/plots/makespan_comparison.png`
- `results/plots/penalized_makespan_comparison.png`
- `results/plots/feasibility_comparison.png`
- `results/plots/runtime_comparison.png`
- `results/plots/route_size_<size>_<method>.png`

Trained PPO models are saved under `models/` as `ppo_n<size>.zip`.
When curriculum training is used, `models/ppo_curriculum.zip` is also saved.

The detailed CSV reports feasibility-first metrics for every method and instance:

- `completion_rate`
- `served_customers`
- `unserved_customers`
- `feasible`
- `invalid_actions`
- `makespan`
- `total_distance`
- `total_reward`
- `runtime_seconds`
- `inference_runtime_seconds`
- `evaluation_runtime_seconds`
- `training_runtime_seconds`
- `total_runtime_seconds`
- `nn_makespan`
- `nn_distance`
- `gap_to_nn_makespan_percent`
- `gap_to_nn_distance_percent`
- `penalized_makespan`
- `terminal_status`
- `route_signature`
- `service_sequence`
- `truck_route`
- `num_unique_actions`

For PPO, per-instance runtime columns report inference and evaluation time only; PPO training runtime is saved in `summary.csv` as `avg_training_runtime_seconds` using the model training metadata sidecar.

The penalized makespan is `makespan + 10000 * unserved_customers + 1000 * invalid_actions`. Raw makespan is not credited as an improvement unless the method is feasible on all evaluated instances.

## How To Run

Create and activate an environment, then install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For the Anaconda base Python used by this project, install with:

```bash
/opt/anaconda3/bin/python -m pip install -r requirements.txt
```

Run the fast verification workflow:

```bash
python experiments.py --quick
```

Run the default study:

```bash
python experiments.py --sizes 10 20 30 40 --train --evaluate
```

By default, multi-size training uses curriculum MaskablePPO: size 10, then 20, then 30, then 40. To force independent per-size training, add `--no-curriculum`.
Each training stage first runs nearest-neighbor behavior cloning for `--bc-epochs` epochs, then continues with MaskablePPO.

Run a quick curriculum smoke test:

```bash
python experiments.py --sizes 10 20 30 40 --train --evaluate --quick-curriculum --test-instances 10 --train-instances 50
```

Run a single configured size:

```bash
python experiments.py --sizes 20 --timesteps 200000
```

Evaluate PPO without masked inference to audit the raw policy's invalid-action behavior:

```bash
python experiments.py --sizes 20 --evaluate --no-mask-ppo-actions
```

When neither `--train` nor `--evaluate` is provided, the runner does both.

## Reproducing Results

The main reproducibility controls are:

- `--seed`: controls synthetic train and test instance generation
- `--train-instances`: number of synthetic training instances per size
- `--test-instances`: number of unseen evaluation instances per size
- `--timesteps`: PPO training budget override
- `--n-envs`: number of vectorized training environments
- `--no-curriculum`: train separate models instead of one progressive policy
- `--quick-curriculum`: use `50,000` timesteps per curriculum stage
- `--bc-epochs`: nearest-neighbor behavior cloning epochs before PPO, default `3`
- `--no-vns`: skip the VNS metaheuristic baseline
- `--vns-max-iterations`: VNS outer-loop budget, default `200`
- `--vns-max-no-improve`: stop VNS after this many non-improving iterations, default `50`
- `--vns-time-limit`: optional VNS per-instance time limit in seconds

Example:

```bash
python experiments.py --sizes 10 20 30 40 --train --evaluate --seed 42
```

The default independent per-size PPO budgets are:

- `n=10`: `100,000`
- `n=20`: `200,000`
- `n=30`: `300,000`
- `n=40`: `400,000`
- `n=50`: `500,000`

The default curriculum PPO budgets are:

- `n=10`: `200,000`
- `n=20`: `400,000`
- `n=30`: `600,000`
- `n=40`: `800,000`
- `n=50`: `1,000,000`

The `--quick` mode uses:

- `n=10`
- `2,048` PPO timesteps
- `5` test instances
- `1` behavior-cloning epoch
- `20` VNS iterations and `5` non-improving iterations

## VNS Baseline

`vns.py` implements a Variable Neighborhood Search baseline. It starts from the existing nearest-neighbor service sequence, then searches feasible route-order permutations with three neighborhoods:

- Swap two customers
- Relocate one customer to another position
- Reverse a subsequence with 2-opt

Every candidate is executed through `TruckDroneEnv`, so the objective is route-dependent and uses the same realized metrics as PPO evaluation. VNS optimizes `(makespan, total_distance)`, keeps only complete routes with zero invalid actions, and uses seeded NumPy randomness for reproducibility.

## Limitations

This first version intentionally keeps the formulation compact:

- One truck and one drone
- Drone serves one node per sortie
- Truck waits during drone service
- No drone battery constraint yet
- No multi-agent coordination yet
- No multi-objective formulation yet
- Synthetic instances only
- SVI is generated for future equity-aware extensions but is not used in the objective

These constraints keep the study focused on a reproducible single-objective makespan formulation before adding richer humanitarian logistics features.
