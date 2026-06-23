"""
train.py
--------
Main training loop for the D3QN indoor rover navigation agent.

Ties together:
    sim_env.py     - Habitat-Sim environment (RoverEnv)
    preprocess.py  - Frame stacking / domain randomization (FrameStack)
    agent.py       - D3QN model + replay buffer (D3QNAgent)

Implements the curriculum described in the design doc:
    Stage 1: Empty room, fixed start/goal      -> learn basic movement
    Stage 2: Room with obstacles               -> avoid collisions
    Stage 3: Multiple rooms                    -> navigate doorways
    Stage 4: Full building layout              -> long-horizon navigation
    Stage 5: Domain randomized                 -> sim-to-real transfer

Usage:
    python train.py --scene path/to/scene.glb --episodes 5000
    python train.py --config configs/rover.yaml
    python train.py --resume checkpoints/d3qn_ep2000.pt
"""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless training runs have no display to render to
import matplotlib.pyplot as plt
import numpy as np
import torch

from sim_env import RoverEnv, RoverEnvConfig, Action
from preprocess import FrameStack
from agent import D3QNAgent, AgentConfig


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Top-level training run settings."""

    scene_path:      str = ""
    episodes:        int = 5_000
    learn_every:     int = 4         # Gradient step every N env steps
    log_every:       int = 10        # Print progress every N episodes
    checkpoint_every: int = 250      # Save model every N episodes
    checkpoint_dir:  str = "checkpoints"
    log_csv_path:    str = "logs/training_log.csv"
    success_plot_path: str = "logs/success_rate.png"
    use_augmentation: bool = True    # Domain randomization during training
    resume_path:     Optional[str] = None
    seed:            int = 42

    # --- Distance curriculum ---
    # Goals sampled anywhere on a large real-world scan from episode 1 are
    # often unreachable in max_steps, drowning the replay buffer in
    # low-signal transitions. Instead, start with nearby goals and expand
    # the max spawn distance once the agent is actually succeeding at the
    # current distance, lifting the cap entirely once it reaches
    # curriculum_final_m (see RoverEnv.set_max_spawn_distance).
    curriculum_enabled:      bool  = True
    curriculum_start_m:      float = 2.5    # initial max spawn distance
    curriculum_step_m:       float = 2.0    # distance added per advancement
    curriculum_final_m:      float = 20.0   # cap is lifted once this is reached
    curriculum_threshold:    float = 0.5    # rolling-100 success rate needed to advance
    curriculum_min_episodes: int   = 100    # episodes of history required before checking
    curriculum_check_every:  int   = 50     # episodes between advancement checks


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: RoverEnv,
    frame_stack: FrameStack,
    agent: D3QNAgent,
    train_cfg: TrainConfig,
    learning_enabled: bool = True,
) -> dict:
    """
    Run a single episode end-to-end: reset, step until done, learn periodically.

    Args:
        env:               The Habitat-Sim wrapper.
        frame_stack:       Maintains the 4-frame temporal stack.
        agent:             The D3QN agent.
        train_cfg:         Training configuration.
        learning_enabled:  If False, just collect experience without gradient
                            steps (useful for buffer warm-up).

    Returns:
        dict with episode_reward, steps, goal_reached, mean_loss
    """
    raw_obs = env.reset()
    state = frame_stack.reset(raw_obs)

    episode_reward = 0.0
    episode_losses = []
    step_count = 0
    goal_reached = False
    collisions = 0

    done = False
    while not done:
        action = agent.select_action(state)
        raw_next_obs, reward, done, info = env.step(action)
        next_state = frame_stack.step(raw_next_obs)

        agent.remember(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        step_count += 1
        goal_reached = info["goal_reached"]
        if info["collision"]:
            collisions += 1

        if learning_enabled and step_count % train_cfg.learn_every == 0:
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

    return {
        "episode_reward": episode_reward,
        "steps":          step_count,
        "goal_reached":   goal_reached,
        "collisions":     collisions,
        "mean_loss":      float(np.mean(episode_losses)) if episode_losses else 0.0,
        "epsilon":        agent.epsilon,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TrainingLogger:
    """
    Writes per-episode metrics to a CSV file and prints a periodic summary
    to stdout. Keeping logging in its own small class avoids cluttering the
    main loop with file-handling code.
    """

    FIELDS = [
        "episode", "reward", "steps", "goal_reached",
        "collisions", "mean_loss", "epsilon", "elapsed_sec",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        is_new = not os.path.exists(csv_path)
        self._file = open(csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if is_new:
            self._writer.writeheader()

        self._recent_rewards = []
        self._recent_goals = []
        self._success_rate_history = []  # (episode, rolling success rate %) for plotting

    def log(self, episode: int, metrics: dict, elapsed_sec: float):
        row = {
            "episode":      episode,
            "reward":       round(metrics["episode_reward"], 4),
            "steps":        metrics["steps"],
            "goal_reached": int(metrics["goal_reached"]),
            "collisions":   metrics["collisions"],
            "mean_loss":    round(metrics["mean_loss"], 6),
            "epsilon":      round(metrics["epsilon"], 4),
            "elapsed_sec":  round(elapsed_sec, 2),
        }
        self._writer.writerow(row)
        self._file.flush()

        self._recent_rewards.append(metrics["episode_reward"])
        self._recent_goals.append(int(metrics["goal_reached"]))
        # Keep a rolling window of the last 100 episodes for summaries
        self._recent_rewards = self._recent_rewards[-100:]
        self._recent_goals = self._recent_goals[-100:]
        self._success_rate_history.append((episode, self.success_rate))

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._recent_goals)) * 100 if self._recent_goals else 0.0

    @property
    def history_len(self) -> int:
        return len(self._recent_goals)

    def print_summary(self, episode: int, total_episodes: int):
        avg_reward = np.mean(self._recent_rewards) if self._recent_rewards else 0.0
        success_rate = np.mean(self._recent_goals) * 100 if self._recent_goals else 0.0
        print(
            f"[{episode:5d}/{total_episodes}] "
            f"avg_reward(last100)={avg_reward:+7.3f}  "
            f"success_rate={success_rate:5.1f}%  "
        )

    def plot_success_rate(self, save_path: str):
        """Save a PNG of the rolling (last-100) success rate over the run."""
        if not self._success_rate_history:
            return

        episodes, rates = zip(*self._success_rate_history)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rates)
        plt.xlabel("Episode")
        plt.ylabel("Success rate, % (rolling last 100 episodes)")
        plt.title("D3QN Training Success Rate")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()
        print(f"[train.py] Success rate plot saved to {save_path}")

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Curriculum state persistence
# ---------------------------------------------------------------------------
# Curriculum progress (the current max spawn distance) lives outside the
# agent checkpoint since it's a property of the training run, not the model.
# Without this, --resume would restart the curriculum at curriculum_start_m
# every time, re-walking already-mastered easy stages instead of continuing
# from where the run left off.

def _curriculum_state_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "curriculum_state.json")


def _save_curriculum_state(checkpoint_dir: str, max_spawn_distance_m: Optional[float]):
    with open(_curriculum_state_path(checkpoint_dir), "w") as f:
        json.dump({"max_spawn_distance_m": max_spawn_distance_m}, f)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(train_cfg: TrainConfig, env_cfg: RoverEnvConfig, agent_cfg: AgentConfig):
    """
    Orchestrates the full training run:
        1. Build environment, frame stack, agent
        2. Optionally resume from checkpoint
        3. Loop over episodes, logging and checkpointing periodically
        4. Clean up on exit (including on Ctrl+C)
    """
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    scene_name = os.path.splitext(os.path.basename(env_cfg.scene_path))[0]
    checkpoint_dir = os.path.join(train_cfg.checkpoint_dir, scene_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    current_max_dist = train_cfg.curriculum_start_m if train_cfg.curriculum_enabled else None
    env_cfg.max_spawn_distance_m = current_max_dist

    env = RoverEnv(config=env_cfg)

    # Size the curriculum's final (pre-uncap) stage to this scene's actual
    # hardest-case goal distance rather than a fixed guess -- a static
    # ceiling that's much smaller than the real scene (e.g. a small-room
    # default applied to a 50m-long building floor) means the curriculum
    # jumps straight from "moderate distance" to "anything in the whole
    # scene" in one step, which is the same unprepared-for-goal-distance
    # problem the curriculum exists to avoid.
    curriculum_final_m = train_cfg.curriculum_final_m
    if train_cfg.curriculum_enabled:
        scene_p95_dist_m = env.estimate_max_geodesic_distance_m()
        curriculum_final_m = max(train_cfg.curriculum_final_m, scene_p95_dist_m * 1.2)

    print("=" * 60)
    print("D3QN Indoor Rover Navigation - Training")
    print("=" * 60)
    print(f"  Scene:       {env_cfg.scene_path}")
    print(f"  Episodes:    {train_cfg.episodes}")
    print(f"  Augment:     {train_cfg.use_augmentation}")
    print(f"  Checkpoints: {checkpoint_dir}")
    if train_cfg.curriculum_enabled:
        print(
            f"  Curriculum:  start={train_cfg.curriculum_start_m}m  "
            f"step={train_cfg.curriculum_step_m}m  "
            f"final={curriculum_final_m:.1f}m (uncapped beyond this; "
            f"p95 geodesic distance ~{scene_p95_dist_m:.1f}m)  "
            f"threshold={train_cfg.curriculum_threshold * 100:.0f}% success"
        )
    else:
        print("  Curriculum:  disabled (goals sampled anywhere on the navmesh)")
    print("=" * 60)

    frame_stack = FrameStack(augment=train_cfg.use_augmentation)
    agent = D3QNAgent(agent_cfg)
    logger = TrainingLogger(train_cfg.log_csv_path)

    start_episode = 1
    if train_cfg.resume_path:
        agent.load(train_cfg.resume_path, eval_mode=False)
        start_episode = agent.episodes + 1
        print(f"  Resuming from episode {start_episode}")

        if train_cfg.curriculum_enabled:
            state_path = _curriculum_state_path(checkpoint_dir)
            if os.path.exists(state_path):
                with open(state_path) as f:
                    current_max_dist = json.load(f)["max_spawn_distance_m"]
                env.set_max_spawn_distance(current_max_dist)
                print(f"  Curriculum resumed at max_spawn_distance={current_max_dist}")
            else:
                print(
                    f"  No curriculum_state.json in {checkpoint_dir} -- "
                    f"restarting curriculum at {current_max_dist}m."
                )

    best_success_rate = 0.0

    if not train_cfg.resume_path:
        print(f"  Warming up replay buffer (target: {agent.cfg.min_buffer_size} transitions)...")
        while len(agent.memory) < agent.cfg.min_buffer_size:
            run_episode(env, frame_stack, agent, train_cfg, learning_enabled=False)
        print(f"  Buffer ready ({len(agent.memory)} transitions). Starting training.")

    try:
        for episode in range(start_episode, train_cfg.episodes + 1):
            t0 = time.time()

            metrics = run_episode(env, frame_stack, agent, train_cfg)
            agent.decay_epsilon()

            elapsed = time.time() - t0
            logger.log(episode, metrics, elapsed)

            if (
                train_cfg.curriculum_enabled
                and current_max_dist is not None
                and episode % train_cfg.curriculum_check_every == 0
                and logger.history_len >= train_cfg.curriculum_min_episodes
                and logger.success_rate / 100.0 >= train_cfg.curriculum_threshold
            ):
                current_max_dist += train_cfg.curriculum_step_m
                if current_max_dist >= curriculum_final_m:
                    current_max_dist = None
                    env.set_max_spawn_distance(None)
                    print(f"[{episode:5d}] Curriculum complete -- goals now sampled with no distance cap.")
                else:
                    env.set_max_spawn_distance(current_max_dist)
                    print(
                        f"[{episode:5d}] Curriculum advance -> max_spawn_distance="
                        f"{current_max_dist:.1f}m (success_rate={logger.success_rate:.1f}%)"
                    )
                _save_curriculum_state(checkpoint_dir, current_max_dist)

            # While the curriculum is still capping goal distance, the rolling
            # success rate reflects an easier task than full deployment and
            # isn't comparable across stages -- e.g. a 65% success rate at a
            # 2.5m cap means almost nothing about performance on an uncapped
            # building floor. Only start tracking "best" once the cap is
            # lifted, so d3qn_best.pt is actually the best checkpoint at the
            # real task.
            curriculum_complete = not train_cfg.curriculum_enabled or current_max_dist is None
            if curriculum_complete and logger.success_rate > best_success_rate:
                best_success_rate = logger.success_rate
                best_path = os.path.join(checkpoint_dir, "d3qn_best.pt")
                agent.save(best_path)

            if episode % train_cfg.log_every == 0:
                logger.print_summary(episode, train_cfg.episodes)

            if episode % train_cfg.checkpoint_every == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir, f"d3qn_ep{episode}.pt"
                )
                agent.save(ckpt_path)

                # Also keep a rolling "latest" checkpoint for easy resume
                latest_path = os.path.join(
                    checkpoint_dir, "d3qn_latest.pt"
                )
                agent.save(latest_path)

    except KeyboardInterrupt:
        print("\n[train.py] Interrupted by user. Saving checkpoint before exit...")
        interrupt_path = os.path.join(
            checkpoint_dir, f"d3qn_interrupted_ep{agent.episodes}.pt"
        )
        agent.save(interrupt_path)

    finally:
        env.close()
        logger.plot_success_rate(train_cfg.success_plot_path)
        logger.close()
        print("\n[train.py] Training run finished. Resources released.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a D3QN agent for indoor rover navigation in Habitat-Sim."
    )
    parser.add_argument(
        "--scene", type=str, required=False,
        help="Path to a Habitat-Sim .glb scene file."
    )
    parser.add_argument(
        "--episodes", type=int, default=5_000,
        help="Number of training episodes."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from."
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable domain randomization (useful for debugging)."
    )
    parser.add_argument(
        "--no-curriculum", action="store_true",
        help="Disable the goal-distance curriculum; sample goals anywhere on the navmesh from episode 1."
    )
    parser.add_argument(
        "--curriculum-threshold", type=float, default=0.5,
        help="Rolling-100 success rate (0-1) required to advance the curriculum's max spawn distance."
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Max steps per episode before timeout."
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory to save model checkpoints."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.scene:
        default_scene = os.path.join(
            os.path.dirname(__file__),
            "..", "habitat-sim", "data",
            "scene_datasets", "habitat-test-scenes", "apartment_1.glb"
        )
        if os.path.exists(default_scene):
            args.scene = default_scene
            print(f"[train.py] No --scene given, using default: {args.scene}")
        else:
            raise SystemExit(
                "No scene provided and no default scene found.\n"
                "Pass one with --scene path/to/scene.glb\n"
                "Or download test scenes:\n"
                "  python -m habitat_sim.utils.datasets_download "
                "--uris habitat_test_scenes --data-path ../habitat-sim/data/"
            )

    train_cfg = TrainConfig(
        scene_path=args.scene,
        episodes=args.episodes,
        checkpoint_dir=args.checkpoint_dir,
        use_augmentation=not args.no_augment,
        curriculum_enabled=not args.no_curriculum,
        curriculum_threshold=args.curriculum_threshold,
        resume_path=args.resume,
        seed=args.seed,
    )

    env_cfg = RoverEnvConfig(
        scene_path=args.scene,
        max_steps=args.max_steps,
    )

    agent_cfg = AgentConfig(
        n_actions=Action.COUNT,
    )

    train(train_cfg, env_cfg, agent_cfg)


if __name__ == "__main__":
    main()