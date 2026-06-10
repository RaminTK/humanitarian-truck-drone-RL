"""V2 PPO reward and ablation experiments.

This runner is intentionally separate from ``experiments.py`` so v2 experiments
write to ``results_v2/`` and ``models_v2/`` without overwriting the v1 baseline
artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import PolicyResult, run_baselines
from evaluate import evaluate_ppo_policy, final_table, load_ppo_model, print_method_comparison, summarize_results
from instance_generator import generate_instances
from train_ppo import train_curriculum_ppo, train_ppo_for_size, training_metadata_path
from truck_drone_env import DEFAULT_REWARD_CONFIG, RewardConfig, TruckDroneEnv
from vns import vns_policy


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PPOVariant:
    name: str
    description: str
    reward_config: RewardConfig
    behavior_clone_epochs: int
    use_curriculum: bool = True
    eval_mask_actions: bool = True
    reuse_model_from: str | None = None
    include_in_ablation: bool = True
    include_in_reward_comparison: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2 PPO improvement and ablation experiments.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 30, 40])
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--train-instances", type=int, default=100)
    parser.add_argument("--test-instances", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=50)
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results_v2")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models_v2")
    parser.add_argument("--skip-training", action="store_true", help="Evaluate existing v2 models only.")
    parser.add_argument("--skip-ortools", action="store_true")
    parser.add_argument("--include-ortools", action="store_true", help="Include optional OR-Tools baseline.")
    parser.add_argument("--vns-max-iterations", type=int, default=200)
    parser.add_argument("--vns-max-no-improve", type=int, default=50)
    parser.add_argument("--vns-time-limit", type=float, default=1.0)
    parser.add_argument(
        "--include-unmasked-eval",
        action="store_true",
        help="Include the unsafe PPO no-action-mask inference audit.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of PPO variant names to run. Defaults to all v2 variants.",
    )
    parser.add_argument("--quick", action="store_true", help="Small smoke test for all v2 outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_quick_defaults(args)
    _validate_args(args)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    variants = _select_variants(
        _build_variants(args.bc_epochs),
        args.variants,
        include_unmasked_eval=args.include_unmasked_eval,
    )
    commands = {
        "sizes": args.sizes,
        "timesteps": args.timesteps,
        "train_instances": args.train_instances,
        "test_instances": args.test_instances,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "max_nodes": args.max_nodes,
        "variants": [variant.name for variant in variants],
    }
    (args.results_dir / "run_config_v2.json").write_text(json.dumps(commands, indent=2) + "\n")

    model_paths_by_variant = _train_or_load_variants(args, variants)
    detailed = _evaluate_v2(args, variants, model_paths_by_variant)
    detailed = _add_reference_gap_columns(detailed, reference_method="nearest_neighbor", prefix="nn")
    detailed = _add_reference_gap_columns(detailed, reference_method="vns", prefix="vns")

    summary = summarize_results(detailed)
    summary = _add_vns_gap_summary(summary, detailed)
    summary = _attach_variant_metadata(summary, variants)
    summary = _attach_variant_training_runtimes(summary, variants, model_paths_by_variant)

    ablation_results = _ablation_results(summary, variants)
    reward_comparison = _reward_comparison(summary, variants)

    detailed.to_csv(args.results_dir / "detailed_results_v2.csv", index=False)
    summary.to_csv(args.results_dir / "summary_v2.csv", index=False)
    ablation_results.to_csv(args.results_dir / "ablation_results.csv", index=False)
    reward_comparison.to_csv(args.results_dir / "ppo_reward_comparison.csv", index=False)

    print("\nV2 summary table:")
    print(final_table(summary).to_string(index=False))
    print_method_comparison(summary)
    print("\nV2 ablation table:")
    print(
        ablation_results[
            [
                "size",
                "method",
                "mean_makespan",
                "mean_total_distance",
                "feasibility_rate",
                "mean_gap_to_nn_makespan_percent",
                "mean_gap_to_vns_makespan_percent",
                "avg_inference_runtime_seconds",
                "avg_training_runtime_seconds",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nSaved v2 results to {args.results_dir}")


def _build_variants(default_bc_epochs: int) -> list[PPOVariant]:
    current = DEFAULT_REWARD_CONFIG
    no_nn_bonus = replace(current, nearest_neighbor_bonus=0.0)
    reduced_regret = replace(current, nearest_neighbor_regret_weight=0.5)
    stronger_final_makespan = replace(current, final_makespan_weight=0.5)
    no_nn_shaping = replace(current, nearest_neighbor_regret_weight=0.0, nearest_neighbor_bonus=0.0)
    bc_warm_less_imitation = replace(current, nearest_neighbor_regret_weight=0.5, nearest_neighbor_bonus=0.0)

    return [
        PPOVariant(
            name="ppo_full_current",
            description="Current v1 PPO reward, curriculum, masking, and nearest-neighbor behavior cloning.",
            reward_config=current,
            behavior_clone_epochs=default_bc_epochs,
            use_curriculum=True,
            include_in_ablation=True,
            include_in_reward_comparison=True,
        ),
        PPOVariant(
            name="ppo_no_curriculum",
            description="Current reward and behavior cloning, trained independently per size.",
            reward_config=current,
            behavior_clone_epochs=default_bc_epochs,
            use_curriculum=False,
            include_in_ablation=True,
        ),
        PPOVariant(
            name="ppo_no_behavior_cloning",
            description="Current reward and curriculum, but no nearest-neighbor behavior cloning.",
            reward_config=current,
            behavior_clone_epochs=0,
            use_curriculum=True,
            include_in_ablation=True,
        ),
        PPOVariant(
            name="ppo_without_nn_reward_shaping",
            description="Curriculum and behavior cloning with nearest-neighbor reward bonus/regret removed.",
            reward_config=no_nn_shaping,
            behavior_clone_epochs=default_bc_epochs,
            use_curriculum=True,
            include_in_ablation=True,
            include_in_reward_comparison=True,
        ),
        PPOVariant(
            name="ppo_without_action_mask_eval",
            description="Safety audit: ppo_full_current evaluated without inference-time action masking.",
            reward_config=current,
            behavior_clone_epochs=default_bc_epochs,
            use_curriculum=True,
            eval_mask_actions=False,
            reuse_model_from="ppo_full_current",
            include_in_ablation=False,
        ),
        PPOVariant(
            name="ppo_no_nn_bonus",
            description="Current reward with nearest-neighbor imitation bonus set to zero.",
            reward_config=no_nn_bonus,
            behavior_clone_epochs=default_bc_epochs,
            use_curriculum=True,
            include_in_ablation=False,
            include_in_reward_comparison=True,
        ),
        PPOVariant(
            name="ppo_reduced_nn_regret",
            description="Current reward with nearest-neighbor regret penalty reduced from 2.0 to 0.5.",
            reward_config=reduced_regret,
            behavior_clone_epochs=default_bc_epochs,
            use_curriculum=True,
            include_in_ablation=False,
            include_in_reward_comparison=True,
        ),
        PPOVariant(
            name="ppo_stronger_final_makespan",
            description="Current reward with stronger final makespan penalty.",
            reward_config=stronger_final_makespan,
            behavior_clone_epochs=default_bc_epochs,
            use_curriculum=True,
            include_in_ablation=False,
            include_in_reward_comparison=True,
        ),
        PPOVariant(
            name="ppo_bc_warm_less_reward_imitation",
            description="Short behavior-cloning warm start with reduced NN regret and no NN bonus.",
            reward_config=bc_warm_less_imitation,
            behavior_clone_epochs=min(1, default_bc_epochs),
            use_curriculum=True,
            include_in_ablation=False,
            include_in_reward_comparison=True,
        ),
    ]


def _train_or_load_variants(args: argparse.Namespace, variants: list[PPOVariant]) -> dict[str, dict[int, Path]]:
    model_paths_by_variant: dict[str, dict[int, Path]] = {}

    for variant in variants:
        if variant.reuse_model_from is not None:
            if variant.reuse_model_from not in model_paths_by_variant:
                raise ValueError(
                    f"{variant.name} reuses {variant.reuse_model_from}, but the source variant was not selected."
                )
            model_paths_by_variant[variant.name] = dict(model_paths_by_variant[variant.reuse_model_from])
            continue

        variant_model_dir = args.model_dir / variant.name
        variant_model_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_training:
            model_paths_by_variant[variant.name] = {
                size: variant_model_dir / f"ppo_n{size}.zip" for size in args.sizes
            }
            continue

        print(f"\nTraining v2 variant {variant.name}: {variant.description}")
        stage_timesteps = {size: int(args.timesteps) for size in args.sizes}
        if variant.use_curriculum and len(args.sizes) > 1:
            model_paths_by_variant[variant.name] = train_curriculum_ppo(
                sizes=args.sizes,
                stage_timesteps=stage_timesteps,
                train_instances=args.train_instances,
                seed=args.seed,
                model_dir=variant_model_dir,
                log_dir=args.results_dir / "logs" / variant.name,
                n_envs=args.n_envs,
                max_nodes=args.max_nodes,
                behavior_clone_epochs=variant.behavior_clone_epochs,
                reward_config=variant.reward_config,
                verbose=0,
            )
        else:
            model_paths_by_variant[variant.name] = {}
            for size in args.sizes:
                model_paths_by_variant[variant.name][size] = train_ppo_for_size(
                    size=size,
                    total_timesteps=int(args.timesteps),
                    train_instances=args.train_instances,
                    seed=args.seed,
                    model_dir=variant_model_dir,
                    log_dir=args.results_dir / "logs" / variant.name,
                    n_envs=args.n_envs,
                    max_nodes=args.max_nodes,
                    behavior_clone_epochs=variant.behavior_clone_epochs,
                    reward_config=variant.reward_config,
                    verbose=0,
                )

    return model_paths_by_variant


def _evaluate_v2(
    args: argparse.Namespace,
    variants: list[PPOVariant],
    model_paths_by_variant: dict[str, dict[int, Path]],
) -> pd.DataFrame:
    detailed_records: list[dict[str, object]] = []
    include_ortools = args.include_ortools and not args.skip_ortools

    for size in args.sizes:
        print(f"\nEvaluating v2 methods for n={size} on {args.test_instances} shared instances...")
        instances = generate_instances(
            n_nodes=size,
            count=args.test_instances,
            seed=args.seed + 100_000 + size,
        )
        loaded_models = _load_variant_models(size, variants, model_paths_by_variant, args.max_nodes)

        for instance_id, instance in enumerate(instances):
            method_results: dict[str, PolicyResult] = run_baselines(
                instance,
                seed=args.seed + (10_000 * size) + instance_id,
                include_ortools=include_ortools,
            )
            method_results["vns"] = vns_policy(
                instance=instance,
                seed=args.seed + (20_000 * size) + instance_id,
                max_nodes=args.max_nodes,
                max_iterations=args.vns_max_iterations,
                max_no_improve=args.vns_max_no_improve,
                time_limit_seconds=args.vns_time_limit,
            )

            for variant in variants:
                model = loaded_models.get(variant.name)
                if model is None:
                    continue
                result = evaluate_ppo_policy(
                    model=model,
                    instance=instance,
                    size=size,
                    max_nodes=args.max_nodes,
                    mask_actions=variant.eval_mask_actions,
                )
                result.method = variant.name
                method_results[variant.name] = result

            for method, result in method_results.items():
                detailed_records.append(result.to_record(size, instance_id, instance.seed))

    return pd.DataFrame(detailed_records)


def _load_variant_models(
    size: int,
    variants: list[PPOVariant],
    model_paths_by_variant: dict[str, dict[int, Path]],
    max_nodes: int,
) -> dict[str, object]:
    loaded = {}
    expected_env = TruckDroneEnv(n_nodes=size, max_nodes=max_nodes)
    for variant in variants:
        model_path = model_paths_by_variant.get(variant.name, {}).get(size)
        if model_path is None or not model_path.exists():
            print(f"Skipping {variant.name} for n={size}: missing model {model_path}.")
            continue
        model = load_ppo_model(model_path)
        if (
            getattr(model.observation_space, "shape", None) != getattr(expected_env.observation_space, "shape", None)
            or getattr(model.action_space, "n", None) != getattr(expected_env.action_space, "n", None)
        ):
            print(f"Skipping {variant.name} for n={size}: model shape does not match current env.")
            continue
        loaded[variant.name] = model
    return loaded


def _add_reference_gap_columns(
    detailed: pd.DataFrame,
    *,
    reference_method: str,
    prefix: str,
) -> pd.DataFrame:
    if detailed.empty:
        return detailed

    detailed = detailed.copy()
    ref_rows = detailed[detailed["method"] == reference_method][
        ["size", "instance_id", "makespan", "total_distance"]
    ].rename(
        columns={
            "makespan": f"{prefix}_makespan",
            "total_distance": f"{prefix}_distance",
        }
    )
    detailed = detailed.merge(ref_rows, on=["size", "instance_id"], how="left")
    detailed[f"gap_to_{prefix}_makespan_percent"] = np.where(
        detailed[f"{prefix}_makespan"] > 0,
        100.0 * (detailed["makespan"] - detailed[f"{prefix}_makespan"]) / detailed[f"{prefix}_makespan"],
        np.nan,
    )
    detailed[f"gap_to_{prefix}_distance_percent"] = np.where(
        detailed[f"{prefix}_distance"] > 0,
        100.0 * (detailed["total_distance"] - detailed[f"{prefix}_distance"]) / detailed[f"{prefix}_distance"],
        np.nan,
    )
    return detailed


def _add_vns_gap_summary(summary: pd.DataFrame, detailed: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    vns_summary = (
        detailed.groupby(["size", "method"], as_index=False)
        .agg(
            mean_gap_to_vns_makespan_percent=("gap_to_vns_makespan_percent", "mean"),
            mean_gap_to_vns_distance_percent=("gap_to_vns_distance_percent", "mean"),
        )
    )
    return summary.merge(vns_summary, on=["size", "method"], how="left")


def _attach_variant_metadata(summary: pd.DataFrame, variants: list[PPOVariant]) -> pd.DataFrame:
    summary = summary.copy()
    descriptions = {variant.name: variant.description for variant in variants}
    summary["variant_description"] = summary["method"].map(descriptions).fillna("")
    summary["method_family"] = np.where(summary["method"].isin(descriptions), "ppo_variant", "baseline")
    return summary


def _attach_variant_training_runtimes(
    summary: pd.DataFrame,
    variants: list[PPOVariant],
    model_paths_by_variant: dict[str, dict[int, Path]],
) -> pd.DataFrame:
    summary = summary.copy()
    variant_by_name = {variant.name: variant for variant in variants}
    for index, row in summary.iterrows():
        method = row["method"]
        if method not in variant_by_name:
            continue

        variant = variant_by_name[method]
        training_source = variant.reuse_model_from or method
        model_path = model_paths_by_variant.get(training_source, {}).get(int(row["size"]))
        if model_path is None:
            continue
        runtime = _read_training_runtime(model_path)
        if runtime > 0:
            summary.at[index, "avg_training_runtime_seconds"] = runtime
    return summary


def _read_training_runtime(model_path: Path) -> float:
    metadata_path = training_metadata_path(model_path)
    if not metadata_path.exists():
        return 0.0
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0.0
    return float(metadata.get("training_runtime_seconds", 0.0))


def _ablation_results(summary: pd.DataFrame, variants: list[PPOVariant]) -> pd.DataFrame:
    ablation_methods = {variant.name for variant in variants if variant.include_in_ablation}
    return summary[summary["method"].isin(ablation_methods)].copy()


def _reward_comparison(summary: pd.DataFrame, variants: list[PPOVariant]) -> pd.DataFrame:
    reward_methods = {variant.name for variant in variants if variant.include_in_reward_comparison}
    return summary[summary["method"].isin(reward_methods)].copy()


def _select_variants(
    variants: list[PPOVariant],
    selected_names: list[str] | None,
    include_unmasked_eval: bool,
) -> list[PPOVariant]:
    if selected_names is None:
        if include_unmasked_eval:
            return variants
        return [variant for variant in variants if variant.name != "ppo_without_action_mask_eval"]

    variant_by_name = {variant.name: variant for variant in variants}
    missing = sorted(set(selected_names) - set(variant_by_name))
    if missing:
        raise ValueError(f"Unknown v2 PPO variants: {missing}")

    selected = [variant_by_name[name] for name in selected_names]
    required_reuse_sources = {
        variant.reuse_model_from for variant in selected if variant.reuse_model_from is not None
    }
    for source in sorted(required_reuse_sources):
        if source not in selected_names:
            selected.insert(0, variant_by_name[source])
    return selected


def _apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return

    args.sizes = [10]
    args.timesteps = min(args.timesteps, 512)
    args.train_instances = min(args.train_instances, 20)
    args.test_instances = min(args.test_instances, 5)
    args.n_envs = 1
    args.bc_epochs = min(args.bc_epochs, 1)
    args.vns_max_iterations = min(args.vns_max_iterations, 20)
    args.vns_max_no_improve = min(args.vns_max_no_improve, 5)
    args.vns_time_limit = min(args.vns_time_limit or 0.25, 0.25)


def _validate_args(args: argparse.Namespace) -> None:
    if any(size <= 0 or size > args.max_nodes for size in args.sizes):
        raise ValueError(f"all sizes must be in [1, {args.max_nodes}]")
    if args.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if args.train_instances <= 0:
        raise ValueError("train-instances must be positive")
    if args.test_instances <= 0:
        raise ValueError("test-instances must be positive")
    if args.n_envs <= 0:
        raise ValueError("n-envs must be positive")
    if args.vns_max_iterations <= 0:
        raise ValueError("vns-max-iterations must be positive")
    if args.vns_max_no_improve <= 0:
        raise ValueError("vns-max-no-improve must be positive")
    if args.vns_time_limit is not None and args.vns_time_limit <= 0:
        raise ValueError("vns-time-limit must be positive")


if __name__ == "__main__":
    main()
