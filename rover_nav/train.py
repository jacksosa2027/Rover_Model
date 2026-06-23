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
import fcntl
import json
import os
import random
import time
from collections import deque
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
    log_dir:         str = "logs"    # actual logs/plots go in log_dir/<scene_name>/, like checkpoint_dir
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

        # Exclusive, non-blocking lock: two unsynchronized processes
        # appending to the same file is almost certainly what produced the
        # NUL-byte corruption found earlier in this project's actual log.
        # Fail fast and clearly instead of silently corrupting the file.
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._file.close()
            raise RuntimeError(
                f"{csv_path} is already locked by another process. Two unsynchronized "
                "training runs writing to the same log file corrupt it -- if you intended "
                "this, you'll need a separate log_dir/checkpoint_dir per run."
            )

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

    def _read_full_history(self):
        """
        Read every episode ever logged to csv_path -- across this process
        and every prior --resume session, since they all append to the same
        file. Plotting from disk instead of in-memory history means the
        plots show the full run, not just whatever this process has seen.

        Tolerates stray NUL bytes and duplicate episode numbers (both seen
        in practice after resuming from an older checkpoint than
        d3qn_latest.pt, which re-logs episode numbers that already exist)
        by de-duplicating on episode number, keeping the last occurrence on
        disk, and returning everything sorted by episode.

        Returns sorted (episodes, rewards, steps, goal_reached) numpy arrays.
        """
        if not os.path.exists(self.csv_path):
            return np.array([]), np.array([]), np.array([]), np.array([])

        with open(self.csv_path, "r", errors="replace") as f:
            content = f.read().replace("\x00", "")

        by_episode = {}
        for row in csv.DictReader(content.splitlines()):
            try:
                by_episode[int(row["episode"])] = row
            except (KeyError, ValueError):
                continue

        episodes = sorted(by_episode)
        rewards = np.array([float(by_episode[e]["reward"]) for e in episodes])
        steps = np.array([int(by_episode[e]["steps"]) for e in episodes])
        goal_reached = np.array([bool(int(by_episode[e]["goal_reached"])) for e in episodes])
        return np.array(episodes), rewards, steps, goal_reached

    def max_logged_episode(self) -> int:
        """Highest episode number already on disk, or 0 if nothing's logged yet."""
        episodes, _, _, _ = self._read_full_history()
        return int(episodes[-1]) if len(episodes) else 0

    def plot_success_rate(self, save_path: str, window: int = 100):
        """Save a PNG of the rolling success rate over the full logged history."""
        episodes, _, _, goal_reached = self._read_full_history()
        if len(episodes) == 0:
            return

        rates = np.empty(len(episodes))
        buf = deque()
        window_sum = 0
        for i, g in enumerate(goal_reached):
            buf.append(g)
            window_sum += g
            if len(buf) > window:
                window_sum -= buf.popleft()
            rates[i] = window_sum / len(buf) * 100

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rates)
        plt.xlabel("Episode")
        plt.ylabel(f"Success rate, % (rolling last {window} episodes)")
        plt.title("D3QN Training Success Rate")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()
        print(f"[train.py] Success rate plot saved to {save_path}")

    def plot_avg_reward(self, save_path: str, bucket_size: int = 100):
        """Save a PNG of average reward over the full logged history, bucketed every bucket_size episodes."""
        episodes, rewards, _, _ = self._read_full_history()
        if len(episodes) == 0:
            return

        bucket_episodes, bucket_avg_rewards = [], []
        for lo in range(0, len(episodes), bucket_size):
            hi = min(lo + bucket_size, len(episodes))
            bucket_episodes.append(episodes[hi - 1])
            bucket_avg_rewards.append(rewards[lo:hi].mean())

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(bucket_episodes, bucket_avg_rewards)
        plt.xlabel("Episode")
        plt.ylabel(f"Avg reward (per {bucket_size}-episode bucket)")
        plt.title("D3QN Training Average Reward")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()
        print(f"[train.py] Average reward plot saved to {save_path}")

    def plot_avg_steps_to_goal(self, save_path: str, bucket_size: int = 100):
        """
        Save a PNG of average steps-to-goal over the full logged history,
        bucketed every bucket_size episodes. Only successful episodes count
        toward the average (steps in a failed/timed-out episode don't mean
        "how long it took to reach the goal") -- buckets with zero
        successes are left as gaps.
        """
        episodes, _, steps, goal_reached = self._read_full_history()
        if len(episodes) == 0:
            return

        bucket_episodes, bucket_avg_steps = [], []
        for lo in range(0, len(episodes), bucket_size):
            hi = min(lo + bucket_size, len(episodes))
            successes = goal_reached[lo:hi]
            bucket_episodes.append(episodes[hi - 1])
            bucket_avg_steps.append(steps[lo:hi][successes].mean() if successes.any() else np.nan)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(bucket_episodes, bucket_avg_steps, marker="o", markersize=3)
        plt.xlabel("Episode")
        plt.ylabel(f"Avg steps to goal, success only (per {bucket_size}-episode bucket)")
        plt.title("D3QN Training Average Steps to Goal")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()
        print(f"[train.py] Average steps-to-goal plot saved to {save_path}")

    def close(self):
        self._file.close()  # also releases the flock acquired in __init__


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

    # Scene-specific, like checkpoint_dir -- a fixed shared path here would
    # mean training a different scene appends into (and silently merges
    # with, once read back for plotting) this scene's history.
    log_dir = os.path.join(train_cfg.log_dir, scene_name)
    os.makedirs(log_dir, exist_ok=True)
    log_csv_path = os.path.join(log_dir, "training_log.csv")
    success_plot_path = os.path.join(log_dir, "success_rate.png")
    reward_plot_path = os.path.join(log_dir, "avg_reward.png")
    steps_plot_path = os.path.join(log_dir, "avg_steps_to_goal.png")

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

    frame_stack = FrameStack(augment=train_cfg.use_augmentation)
    agent = D3QNAgent(agent_cfg)
    logger = TrainingLogger(log_csv_path)

    start_episode = 1
    curriculum_state_note = None
    if train_cfg.resume_path:
        agent.load(train_cfg.resume_path, eval_mode=False)
        start_episode = agent.episodes + 1

        max_logged = logger.max_logged_episode()
        if max_logged == 0 and agent.episodes > 0:
            print(
                f"  WARNING: resuming from episode {agent.episodes}, but {log_csv_path} "
                "has no logged history. The log may have been deleted or moved -- "
                "plots will be missing everything before this run."
            )
        elif max_logged >= start_episode:
            print(
                f"  WARNING: {log_csv_path} already has entries up to episode {max_logged}, "
                f"but this run resumes from episode {start_episode} -- "
                f"'{train_cfg.resume_path}' is behind the most recently logged progress. "
                f"Episodes {start_episode}-{max_logged} will be OVERWRITTEN in the log/plots "
                "with this run's new trajectory as it catches back up, diverging from "
                "whatever produced the data that's there now."
            )
            try:
                response = input("  Type 'yes' to continue anyway, or anything else to abort: ").strip().lower()
            except EOFError:
                response = ""
            if response != "yes":
                raise SystemExit(
                    "[train.py] Aborted -- resume checkpoint is behind already-logged progress."
                )

        if train_cfg.curriculum_enabled:
            state_path = _curriculum_state_path(checkpoint_dir)
            if os.path.exists(state_path):
                with open(state_path) as f:
                    current_max_dist = json.load(f)["max_spawn_distance_m"]
                env.set_max_spawn_distance(current_max_dist)
                curriculum_state_note = f"resumed from {state_path}"
            else:
                curriculum_state_note = f"no {state_path} found -- restarting curriculum"

    # Printed after resume so it reflects what's actually in effect for the
    # next episode, not the schedule's static defaults -- e.g. on resume,
    # current_max_dist may already be far past curriculum_start_m.
    print("=" * 60)
    print("D3QN Indoor Rover Navigation - Training")
    print("=" * 60)
    print(f"  Scene:       {env_cfg.scene_path}")
    print(f"  Episodes:    {train_cfg.episodes}  (starting at {start_episode})")
    print(f"  Augment:     {train_cfg.use_augmentation}")
    print(f"  Checkpoints: {checkpoint_dir}")
    if train_cfg.curriculum_enabled:
        current_dist_label = "uncapped" if current_max_dist is None else f"{current_max_dist:.1f}m"
        print(
            f"  Curriculum:  current={current_dist_label}  "
            f"step={train_cfg.curriculum_step_m}m  "
            f"final={curriculum_final_m:.1f}m (uncapped beyond this; "
            f"p95 geodesic distance ~{scene_p95_dist_m:.1f}m)  "
            f"threshold={train_cfg.curriculum_threshold * 100:.0f}% success"
        )
        if curriculum_state_note:
            print(f"               ({curriculum_state_note})")
    else:
        print("  Curriculum:  disabled (goals sampled anywhere on the navmesh)")
    print("=" * 60)

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
        logger.plot_success_rate(success_plot_path)
        logger.plot_avg_reward(reward_plot_path)
        logger.plot_avg_steps_to_goal(steps_plot_path)
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