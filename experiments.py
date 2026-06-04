"""Command-line experiment runner for the truck-drone PPO study."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate import evaluate_size, final_table
from instance_generator import INSTANCE_SIZES, generate_instances
from train_ppo import (
    DEFAULT_CURRICULUM_TIMESTEPS,
    DEFAULT_TIMESTEPS,
    QUICK_CURRICULUM_TIMESTEPS,
    train_curriculum_ppo,
    train_ppo_for_size,
)
from visualize import (
    plot_combined_learning_curves,
    plot_learning_curve,
    plot_metric_bars,
    plot_route,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPO and baseline experiments for humanitarian truck-drone routing."
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 30, 40])
    parser.add_argument("--train", action="store_true", help="Train PPO models.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate PPO and baselines.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override PPO timesteps for all sizes.")
    parser.add_argument("--train-instances", type=int, default=500)
    parser.add_argument("--test-instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=50)
    parser.add_argument("--bc-epochs", type=int, default=3, help="Nearest-neighbor behavior cloning epochs before PPO.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--skip-ortools", action="store_true", help="Skip the optional OR-Tools baseline.")
    parser.add_argument(
        "--no-mask-ppo-actions",
        action="store_true",
        help="Evaluate PPO without the environment valid-action mask.",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Train separate models instead of progressive curriculum training.",
    )
    parser.add_argument(
        "--quick-curriculum",
        action="store_true",
        help="Use 50000 PPO timesteps per curriculum stage.",
    )
    parser.add_argument("--quick", action="store_true", help="Run n=10, 2048 PPO steps, 5 tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_defaults(args)
    _validate_args(args)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_root = args.results_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    all_summary: list[pd.DataFrame] = []
    all_detailed: list[pd.DataFrame] = []
    trained_model_paths: dict[int, Path] = {}

    use_curriculum = args.train and not args.no_curriculum and len(args.sizes) > 1
    if use_curriculum:
        stage_timesteps = _curriculum_timesteps(args)
        trained_model_paths = train_curriculum_ppo(
            sizes=args.sizes,
            stage_timesteps=stage_timesteps,
            train_instances=args.train_instances,
            seed=args.seed,
            model_dir=args.model_dir,
            log_dir=log_root,
            n_envs=args.n_envs,
            max_nodes=args.max_nodes,
            behavior_clone_epochs=args.bc_epochs,
        )
        for size in args.sizes:
            plot_learning_curve(
                log_root / f"n_{size}",
                plots_dir / f"learning_curve_n{size}.png",
                title=f"MaskablePPO curriculum learning curve, n={size}",
            )

    for size in args.sizes:
        model_path = trained_model_paths.get(size, args.model_dir / f"ppo_n{size}.zip")
        timesteps = _timesteps_for_size(args, size)

        if args.train and not use_curriculum:
            print(f"\nTraining PPO for n={size} with {timesteps:,} timesteps...")
            model_path = train_ppo_for_size(
                size=size,
                total_timesteps=timesteps,
                train_instances=args.train_instances,
                seed=args.seed,
                model_dir=args.model_dir,
                log_dir=log_root,
                n_envs=args.n_envs,
                max_nodes=args.max_nodes,
                behavior_clone_epochs=args.bc_epochs,
            )
            plot_learning_curve(
                log_root / f"n_{size}",
                plots_dir / f"learning_curve_n{size}.png",
                title=f"PPO learning curve, n={size}",
            )

        if args.evaluate:
            print(f"Evaluating methods for n={size} on {args.test_instances} instances...")
            test_instances = generate_instances(
                n_nodes=size,
                count=args.test_instances,
                seed=args.seed + 100_000 + size,
            )
            summary, detailed, traces = evaluate_size(
                size=size,
                instances=test_instances,
                model_path=model_path if model_path.exists() else None,
                seed=args.seed,
                include_ortools=not args.skip_ortools,
                max_nodes=args.max_nodes,
                mask_ppo_actions=not args.no_mask_ppo_actions,
            )
            all_summary.append(summary)
            all_detailed.append(detailed)

            if test_instances:
                for method, trace in traces.items():
                    plot_route(
                        test_instances[0],
                        trace,
                        plots_dir / f"route_size_{size}_{method}.png",
                        title=f"Route visualization, n={size}, {method}",
                    )

    if all_detailed:
        detailed_results = pd.concat(all_detailed, ignore_index=True)
        summary_results = pd.concat(all_summary, ignore_index=True)
        summary_results = summary_results.sort_values(
            ["size", "feasibility_rate", "completion_rate", "mean_penalized_makespan", "method"],
            ascending=[True, False, False, True, True],
        )

        detailed_results.to_csv(args.results_dir / "detailed_results.csv", index=False)
        summary_results.to_csv(args.results_dir / "summary.csv", index=False)

        plot_metric_bars(
            summary_results,
            metric="mean_makespan",
            output_path=plots_dir / "makespan_comparison.png",
            title="Raw mean makespan comparison",
            ylabel="Raw mean makespan",
        )
        plot_metric_bars(
            summary_results,
            metric="mean_penalized_makespan",
            output_path=plots_dir / "penalized_makespan_comparison.png",
            title="Feasibility-penalized makespan comparison",
            ylabel="Mean penalized makespan",
        )
        plot_metric_bars(
            summary_results,
            metric="feasibility_rate",
            output_path=plots_dir / "feasibility_comparison.png",
            title="Feasibility rate comparison",
            ylabel="Feasibility rate",
        )
        plot_metric_bars(
            summary_results,
            metric="mean_runtime",
            output_path=plots_dir / "runtime_comparison.png",
            title="Mean runtime comparison",
            ylabel="Runtime per instance (seconds)",
        )
        plot_combined_learning_curves(log_root, plots_dir / "convergence_reward_curve.png")

        print("\nFinal summary table:")
        print(final_table(summary_results).to_string(index=False))
        print(f"\nSaved results to {args.results_dir}")
    elif args.train:
        plot_combined_learning_curves(log_root, plots_dir / "convergence_reward_curve.png")
        print(f"\nTraining complete. Models saved to {args.model_dir}")


def _apply_defaults(args: argparse.Namespace) -> None:
    if args.quick:
        args.sizes = [10]
        args.train = True
        args.evaluate = True
        args.timesteps = 2_048
        args.test_instances = min(args.test_instances, 5)
        args.train_instances = min(args.train_instances, 20)
        args.n_envs = 1
        args.bc_epochs = min(args.bc_epochs, 1)
        args.skip_ortools = True

    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True


def _validate_args(args: argparse.Namespace) -> None:
    invalid_sizes = [size for size in args.sizes if size <= 0 or size > args.max_nodes]
    if invalid_sizes:
        raise ValueError(f"sizes must be in [1, {args.max_nodes}], got {invalid_sizes}")

    unsupported_defaults = [size for size in args.sizes if size not in INSTANCE_SIZES]
    if unsupported_defaults:
        print(f"Warning: using custom sizes outside documented defaults: {unsupported_defaults}")

    if args.train_instances <= 0:
        raise ValueError("train-instances must be positive")
    if args.test_instances <= 0:
        raise ValueError("test-instances must be positive")
    if args.n_envs <= 0:
        raise ValueError("n-envs must be positive")


def _timesteps_for_size(args: argparse.Namespace, size: int) -> int:
    if args.timesteps is not None:
        return int(args.timesteps)
    return int(DEFAULT_TIMESTEPS.get(size, 100_000))


def _curriculum_timesteps(args: argparse.Namespace) -> dict[int, int]:
    if args.timesteps is not None:
        return {size: int(args.timesteps) for size in args.sizes}
    if args.quick_curriculum:
        return {size: int(QUICK_CURRICULUM_TIMESTEPS.get(size, 50_000)) for size in args.sizes}
    return {size: int(DEFAULT_CURRICULUM_TIMESTEPS.get(size, 200_000)) for size in args.sizes}


if __name__ == "__main__":
    main()
