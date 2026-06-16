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
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional

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
    use_augmentation: bool = True    # Domain randomization during training
    resume_path:     Optional[str] = None
    seed:            int = 42


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

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._recent_goals)) * 100 if self._recent_goals else 0.0

    def print_summary(self, episode: int, total_episodes: int):
        avg_reward = np.mean(self._recent_rewards) if self._recent_rewards else 0.0
        success_rate = np.mean(self._recent_goals) * 100 if self._recent_goals else 0.0
        print(
            f"[{episode:5d}/{total_episodes}] "
            f"avg_reward(last100)={avg_reward:+7.3f}  "
            f"success_rate={success_rate:5.1f}%  "
        )

    def close(self):
        self._file.close()


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

    print("=" * 60)
    print("D3QN Indoor Rover Navigation - Training")
    print("=" * 60)
    print(f"  Scene:       {env_cfg.scene_path}")
    print(f"  Episodes:    {train_cfg.episodes}")
    print(f"  Augment:     {train_cfg.use_augmentation}")
    print("=" * 60)

    env = RoverEnv(config=env_cfg)
    frame_stack = FrameStack(augment=train_cfg.use_augmentation)
    agent = D3QNAgent(agent_cfg)
    logger = TrainingLogger(train_cfg.log_csv_path)

    start_episode = 1
    if train_cfg.resume_path:
        agent.load(train_cfg.resume_path, eval_mode=False)
        start_episode = agent.episodes + 1
        print(f"  Resuming from episode {start_episode}")

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

            if logger.success_rate > best_success_rate:
                best_success_rate = logger.success_rate
                best_path = os.path.join(train_cfg.checkpoint_dir, "d3qn_best.pt")
                agent.save(best_path)

            if episode % train_cfg.log_every == 0:
                logger.print_summary(episode, train_cfg.episodes)

            if episode % train_cfg.checkpoint_every == 0:
                ckpt_path = os.path.join(
                    train_cfg.checkpoint_dir, f"d3qn_ep{episode}.pt"
                )
                agent.save(ckpt_path)

                # Also keep a rolling "latest" checkpoint for easy resume
                latest_path = os.path.join(
                    train_cfg.checkpoint_dir, "d3qn_latest.pt"
                )
                agent.save(latest_path)

    except KeyboardInterrupt:
        print("\n[train.py] Interrupted by user. Saving checkpoint before exit...")
        interrupt_path = os.path.join(
            train_cfg.checkpoint_dir, f"d3qn_interrupted_ep{agent.episodes}.pt"
        )
        agent.save(interrupt_path)

    finally:
        env.close()
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