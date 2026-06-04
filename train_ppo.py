"""MaskablePPO training utilities for truck-drone routing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from instance_generator import generate_instances
from truck_drone_env import TruckDroneEnv
from baselines import nearest_neighbor_policy


INSTALL_COMMAND = "/opt/anaconda3/bin/python -m pip install -r requirements.txt"
DEFAULT_TIMESTEPS = {
    10: 100_000,
    20: 200_000,
    30: 300_000,
    40: 400_000,
    50: 500_000,
}

DEFAULT_CURRICULUM_TIMESTEPS = {
    10: 200_000,
    20: 400_000,
    30: 600_000,
    40: 800_000,
    50: 1_000_000,
}

QUICK_CURRICULUM_TIMESTEPS = {
    10: 50_000,
    20: 50_000,
    30: 50_000,
    40: 50_000,
    50: 50_000,
}


def train_ppo_for_size(
    size: int,
    total_timesteps: int | None = None,
    train_instances: int = 500,
    seed: int = 42,
    model_dir: str | Path = "models",
    log_dir: str | Path = "results/logs",
    n_envs: int = 4,
    max_nodes: int = 50,
    behavior_clone_epochs: int = 3,
    verbose: int = 1,
) -> Path:
    """Train and save one MaskablePPO model for a fixed instance size."""

    (
        BaseCallback,
        Monitor,
        DummyVecEnv,
        MaskablePPO,
        get_action_masks,
        is_masking_supported,
    ) = _load_maskable_ppo_dependencies()

    total_timesteps = int(total_timesteps or DEFAULT_TIMESTEPS.get(size, 100_000))
    model_dir = Path(model_dir)
    log_dir = Path(log_dir) / f"n_{size}"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    instance_pool = generate_instances(
        n_nodes=size,
        count=train_instances,
        seed=seed + size,
    )

    env = _make_vec_env(
        size=size,
        max_nodes=max_nodes,
        instance_pool=instance_pool,
        seed=seed,
        n_envs=n_envs,
        log_dir=log_dir,
        Monitor=Monitor,
        DummyVecEnv=DummyVecEnv,
    )
    _validate_maskable_env(
        env,
        get_action_masks=get_action_masks,
        is_masking_supported=is_masking_supported,
    )
    rollout_steps = _rollout_steps(total_timesteps, n_envs)
    batch_size = _batch_size(rollout_steps, n_envs)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=rollout_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        verbose=verbose,
        seed=seed,
    )
    if behavior_clone_epochs > 0:
        _behavior_clone_from_nearest_neighbor(
            model=model,
            instances=instance_pool,
            size=size,
            max_nodes=max_nodes,
            epochs=behavior_clone_epochs,
            verbose=verbose,
        )
    validation_instances = generate_instances(
        n_nodes=size,
        count=min(20, max(5, train_instances // 10)),
        seed=seed + 200_000 + size,
    )
    callback = _make_training_callback(
        BaseCallback=BaseCallback,
        validation_instances=validation_instances,
        size=size,
        max_nodes=max_nodes,
        eval_freq=max(rollout_steps, total_timesteps // 5),
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    output_path = model_dir / f"ppo_n{size}.zip"
    model.save(output_path)
    env.close()
    return output_path


def train_curriculum_ppo(
    sizes: list[int],
    stage_timesteps: dict[int, int] | None = None,
    train_instances: int = 500,
    seed: int = 42,
    model_dir: str | Path = "models",
    log_dir: str | Path = "results/logs",
    n_envs: int = 4,
    max_nodes: int = 50,
    behavior_clone_epochs: int = 3,
    verbose: int = 1,
) -> dict[int, Path]:
    """Train one fixed-action MaskablePPO policy progressively across sizes."""

    (
        BaseCallback,
        Monitor,
        DummyVecEnv,
        MaskablePPO,
        get_action_masks,
        is_masking_supported,
    ) = _load_maskable_ppo_dependencies()

    model_dir = Path(model_dir)
    log_root = Path(log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    model = None
    saved_paths: dict[int, Path] = {}
    stage_timesteps = stage_timesteps or DEFAULT_CURRICULUM_TIMESTEPS

    for stage_index, size in enumerate(sizes):
        total_timesteps = int(stage_timesteps.get(size, DEFAULT_CURRICULUM_TIMESTEPS.get(size, 200_000)))
        print(f"\nCurriculum stage {stage_index + 1}/{len(sizes)}: n={size}, timesteps={total_timesteps:,}")

        instance_pool = generate_instances(
            n_nodes=size,
            count=train_instances,
            seed=seed + size,
        )
        stage_log_dir = log_root / f"n_{size}"
        stage_log_dir.mkdir(parents=True, exist_ok=True)
        env = _make_vec_env(
            size=size,
            max_nodes=max_nodes,
            instance_pool=instance_pool,
            seed=seed + (10_000 * stage_index),
            n_envs=n_envs,
            log_dir=stage_log_dir,
            Monitor=Monitor,
            DummyVecEnv=DummyVecEnv,
        )
        _validate_maskable_env(
            env,
            get_action_masks=get_action_masks,
            is_masking_supported=is_masking_supported,
        )
        rollout_steps = _rollout_steps(total_timesteps, n_envs)
        batch_size = _batch_size(rollout_steps, n_envs)

        if model is None:
            model = MaskablePPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=1e-4,
                n_steps=rollout_steps,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                verbose=verbose,
                seed=seed,
            )
        else:
            if not _spaces_match(model, env):
                env.close()
                raise RuntimeError(
                    "Curriculum stages must use identical observation and action spaces. "
                    f"Use one fixed max_nodes value; current model obs={model.observation_space}, "
                    f"action={model.action_space}, new env obs={env.observation_space}, "
                    f"action={env.action_space}."
                )
            if verbose and (rollout_steps != model.n_steps or batch_size != model.batch_size):
                print(
                    "Keeping the original rollout buffer shape for curriculum reuse "
                    f"(n_steps={model.n_steps}, batch_size={model.batch_size})."
                )
            model.set_env(env)

        if behavior_clone_epochs > 0:
            _behavior_clone_from_nearest_neighbor(
                model=model,
                instances=instance_pool,
                size=size,
                max_nodes=max_nodes,
                epochs=behavior_clone_epochs,
                verbose=verbose,
            )

        validation_instances = generate_instances(
            n_nodes=size,
            count=min(20, max(5, train_instances // 10)),
            seed=seed + 200_000 + size,
        )
        callback = _make_training_callback(
            BaseCallback=BaseCallback,
            validation_instances=validation_instances,
            size=size,
            max_nodes=max_nodes,
            eval_freq=max(rollout_steps, total_timesteps // 5),
            verbose=verbose,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=stage_index == 0,
            progress_bar=False,
        )

        output_path = model_dir / f"ppo_n{size}.zip"
        model.save(output_path)
        saved_paths[size] = output_path
        env.close()

    if model is not None:
        model.save(model_dir / "ppo_curriculum.zip")

    return saved_paths


def _load_maskable_ppo_dependencies():
    try:
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            "MaskablePPO training requires stable-baselines3, sb3-contrib, gymnasium, "
            f"numpy, pandas, matplotlib, and torch. Missing import: {missing!r}. "
            f"Install project dependencies with: {INSTALL_COMMAND}"
        ) from exc
    return BaseCallback, Monitor, DummyVecEnv, MaskablePPO, get_action_masks, is_masking_supported


def _make_vec_env(
    *,
    size: int,
    max_nodes: int,
    instance_pool: list,
    seed: int,
    n_envs: int,
    log_dir: Path,
    Monitor: object,
    DummyVecEnv: object,
):
    def make_env(rank: int):
        def _init():
            env = TruckDroneEnv(
                n_nodes=size,
                max_nodes=max_nodes,
                instance_pool=instance_pool,
                seed=seed + (1000 * rank),
            )
            env = Monitor(
                env,
                filename=str(log_dir / f"env_{rank}"),
                info_keywords=(
                    "completion_rate",
                    "feasible",
                    "served_customers",
                    "unserved_customers",
                    "invalid_actions",
                    "makespan",
                    "total_distance",
                    "average_regret",
                    "penalized_makespan",
                    "terminal_status",
                ),
            )
            return env

        return _init

    return DummyVecEnv([make_env(rank) for rank in range(max(1, n_envs))])


def _validate_maskable_env(
    env,
    *,
    get_action_masks,
    is_masking_supported,
) -> None:
    if not is_masking_supported(env):
        raise RuntimeError(
            "MaskablePPO environment does not expose action_masks(). "
            "TruckDroneEnv must provide a 1D boolean mask of length action_space.n."
        )

    masks = np.asarray(get_action_masks(env), dtype=bool)
    expected_shape = (env.num_envs, env.action_space.n)
    if masks.shape != expected_shape:
        raise RuntimeError(
            f"Invalid action mask shape {masks.shape}; expected {expected_shape}."
        )
    if not masks.any(axis=1).all():
        raise RuntimeError("Invalid action mask: every vectorized env must expose at least one valid action.")


def _spaces_match(model, env) -> bool:
    return (
        getattr(model.observation_space, "shape", None) == getattr(env.observation_space, "shape", None)
        and getattr(model.action_space, "n", None) == getattr(env.action_space, "n", None)
    )


def _rollout_steps(total_timesteps: int, n_envs: int) -> int:
    return max(64, min(2048, total_timesteps // max(1, n_envs)))


def _batch_size(rollout_steps: int, n_envs: int) -> int:
    return min(256, rollout_steps * max(1, n_envs))


def _make_training_callback(
    *,
    BaseCallback: object,
    validation_instances: list,
    size: int,
    max_nodes: int,
    eval_freq: int,
    verbose: int,
):
    class TrainingDiagnosticsCallback(BaseCallback):
        """Log feasibility and heuristic-gap metrics at episode and validation boundaries."""

        def __init__(self) -> None:
            super().__init__(verbose=verbose)
            self.episode_count = 0
            self.completions: list[float] = []
            self.feasible: list[float] = []
            self.invalid_actions: list[float] = []
            self.unserved_customers: list[float] = []
            self.makespans: list[float] = []
            self.distances: list[float] = []
            self.regrets: list[float] = []

        def _on_step(self) -> bool:
            dones = self.locals.get("dones", [])
            infos = self.locals.get("infos", [])

            for done, info in zip(dones, infos):
                if not done:
                    continue
                self.episode_count += 1
                self.completions.append(float(info.get("completion_rate", 0.0)))
                self.feasible.append(float(info.get("feasible", False)))
                self.invalid_actions.append(float(info.get("invalid_actions", 0.0)))
                self.unserved_customers.append(float(info.get("unserved_customers", 0.0)))
                self.makespans.append(float(info.get("makespan", 0.0)))
                self.distances.append(float(info.get("total_distance", 0.0)))
                self.regrets.append(float(info.get("average_regret", 0.0)))

                self.logger.record("rollout/completion_rate", self.completions[-1])
                self.logger.record("rollout/feasible", self.feasible[-1])
                self.logger.record("rollout/invalid_actions", self.invalid_actions[-1])
                self.logger.record("rollout/unserved_customers", self.unserved_customers[-1])
                self.logger.record("rollout/makespan", self.makespans[-1])
                self.logger.record("rollout/total_distance", self.distances[-1])
                self.logger.record("rollout/average_regret", self.regrets[-1])

            if self.num_timesteps % eval_freq < self.training_env.num_envs:
                metrics = _evaluate_model_against_nn(
                    model=self.model,
                    instances=validation_instances,
                    size=size,
                    max_nodes=max_nodes,
                )
                for key, value in metrics.items():
                    self.logger.record(f"validation/{key}", value)
                if self.verbose:
                    print(
                        "Validation "
                        f"timesteps={self.num_timesteps} "
                        f"completion={metrics['completion_rate']:.3f} "
                        f"invalid={metrics['invalid_actions']:.2f} "
                        f"ppo_makespan={metrics['ppo_makespan']:.2f} "
                        f"ppo_distance={metrics['ppo_distance']:.2f} "
                        f"nn_makespan={metrics['nearest_neighbor_makespan']:.2f} "
                        f"nn_distance={metrics['nearest_neighbor_distance']:.2f} "
                        f"gap={metrics['gap_to_nearest_neighbor_percent']:.2f}%"
                    )

            return True

    return TrainingDiagnosticsCallback()


def _evaluate_model_against_nn(
    *,
    model,
    instances: list,
    size: int,
    max_nodes: int,
) -> dict[str, float]:
    ppo_makespans: list[float] = []
    ppo_distances: list[float] = []
    completions: list[float] = []
    invalid_actions: list[float] = []
    nn_makespans: list[float] = []
    nn_distances: list[float] = []

    for instance in instances:
        env = TruckDroneEnv(n_nodes=size, max_nodes=max_nodes)
        obs, _ = env.reset(options={"instance": instance})
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=True,
            )
            obs, _, terminated, truncated, info = env.step(int(np.asarray(action).item()))

        nn_result = nearest_neighbor_policy(instance)
        ppo_makespans.append(float(info.get("makespan", env.elapsed_time)))
        ppo_distances.append(float(info.get("total_distance", env.total_distance)))
        completions.append(float(info.get("completion_rate", 0.0)))
        invalid_actions.append(float(info.get("invalid_actions", 0.0)))
        nn_makespans.append(nn_result.makespan)
        nn_distances.append(nn_result.total_distance)

    ppo_makespan = float(np.mean(ppo_makespans))
    nn_makespan = float(np.mean(nn_makespans))
    ppo_distance = float(np.mean(ppo_distances))
    nn_distance = float(np.mean(nn_distances))
    return {
        "completion_rate": float(np.mean(completions)),
        "invalid_actions": float(np.mean(invalid_actions)),
        "ppo_makespan": ppo_makespan,
        "ppo_distance": ppo_distance,
        "nearest_neighbor_makespan": nn_makespan,
        "nearest_neighbor_distance": nn_distance,
        "gap_to_nearest_neighbor_percent": (
            100.0 * (ppo_makespan - nn_makespan) / nn_makespan if nn_makespan else np.nan
        ),
    }


def _behavior_clone_from_nearest_neighbor(
    *,
    model,
    instances: list,
    size: int,
    max_nodes: int,
    epochs: int,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    verbose: int = 0,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"torch is required for behavior cloning. Install project dependencies with: {INSTALL_COMMAND}"
        ) from exc

    observations: list[np.ndarray] = []
    actions: list[int] = []
    for instance in instances:
        expert = nearest_neighbor_policy(instance)
        env = TruckDroneEnv(n_nodes=size, max_nodes=max_nodes)
        obs, _ = env.reset(options={"instance": instance})
        for action in expert.service_sequence:
            observations.append(obs.copy())
            actions.append(int(action))
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break

    if not observations:
        return

    obs_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.int64)
    rng = np.random.default_rng(12345 + size)
    model.policy.set_training_mode(True)
    original_lrs = [group["lr"] for group in model.policy.optimizer.param_groups]
    for group in model.policy.optimizer.param_groups:
        group["lr"] = learning_rate

    for epoch in range(epochs):
        indices = rng.permutation(len(action_array))
        losses: list[float] = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            obs_tensor = torch.as_tensor(obs_array[batch_indices], device=model.device).float()
            action_tensor = torch.as_tensor(action_array[batch_indices], device=model.device).long()

            distribution = model.policy.get_distribution(obs_tensor)
            logits = distribution.distribution.logits
            mask_start = 2 + max_nodes + max_nodes + (2 * max_nodes) + (3 * max_nodes)
            mask_end = mask_start + max_nodes + 1
            action_mask = obs_tensor[:, mask_start:mask_end].bool()
            masked_logits = logits.masked_fill(~action_mask, -1e9)
            loss = torch.nn.functional.cross_entropy(masked_logits, action_tensor)

            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        if verbose:
            print(
                f"Behavior cloning n={size} epoch={epoch + 1}/{epochs} "
                f"samples={len(action_array)} loss={np.mean(losses):.4f}"
            )

    for group, original_lr in zip(model.policy.optimizer.param_groups, original_lrs):
        group["lr"] = original_lr
