"""
evaluate.py
-----------
Evaluate a trained D3QN checkpoint on the indoor rover navigation task.

Runs N episodes with epsilon=0 (fully greedy policy, no exploration) and reports:
    success_rate     - % of episodes where the goal was reached  ("accuracy")
    collision_rate    - % of episodes with at least one collision
    timeout_rate      - % of episodes that hit max_steps without reaching goal
    avg_reward        - mean episode reward
    avg_steps_to_goal - mean steps per episode, successful episodes only

Note: classification-style "recall" isn't well-defined for a single binary
per-episode outcome (goal reached / not reached), so this reports
collision_rate and timeout_rate alongside success_rate as the closest
safety/efficiency analogues.

Usage:
    python rover_nav/evaluate.py \
        --checkpoint checkpoints/ECE_floor/d3qn_best.pt \
        --scene /home/user/data/habitat-sim/versioned_data/habitat_test_scenes/ECE_floor.glb \
        --episodes 100 \
        --max-spawn-distance 14.0

Distance sweep (success rate as a function of spawn distance, plotted and
saved to a PNG -- goals are sampled uncapped across the full scene so the
sweep isn't limited to one curriculum stage):
    python rover_nav/evaluate.py \
        --checkpoint checkpoints/ECE_floor/d3qn_latest.pt \
        --scene /home/user/data/habitat-sim/versioned_data/habitat_test_scenes/ECE_floor.glb \
        --episodes 300 \
        --distance-sweep
"""

import argparse
import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless -- no display to render to
import matplotlib.pyplot as plt

from sim_env import RoverEnv, RoverEnvConfig, Action
from preprocess import FrameStack
from agent import D3QNAgent, AgentConfig


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

# A pure-greedy (epsilon=0) DQN policy has no noise to break ties, so it can
# lock into a repeating action cycle -- e.g. turning in place or nosing into
# the same wall forever. If the geodesic distance to goal hasn't moved by
# more than STUCK_DIST_EPS over STUCK_WINDOW steps, we inject a random turn
# to break the cycle. After the turn, recent_dists is cleared so the policy
# gets STUCK_WINDOW fresh steps with the new heading before re-triggering.
# FORWARD was previously added as a second escape step to change position,
# but it caused the agent to nose into walls on almost every escape (89%
# collision rate) since the agent is usually stuck because there is a wall
# directly in front of it. Turn-only avoids that.
STUCK_WINDOW = 20
STUCK_DIST_EPS = 0.05  # meters
STUCK_ESCAPE_ACTIONS = (Action.LEFT, Action.RIGHT)


def run_eval_episode(env: RoverEnv, frame_stack: FrameStack, agent: D3QNAgent) -> dict:
    """Run one episode greedily (no learning, minimal exploration) and return its outcome."""
    raw_obs = env.reset()
    spawn_distance = env.get_spawn_distance()
    state = frame_stack.reset(raw_obs)

    episode_reward = 0.0
    steps = 0
    collisions = 0
    goal_reached = False
    timeout = False
    stuck_escapes = 0
    recent_dists: deque = deque(maxlen=STUCK_WINDOW)

    done = False
    while not done:
        stuck = (
            len(recent_dists) == STUCK_WINDOW
            and max(recent_dists) - min(recent_dists) < STUCK_DIST_EPS
        )
        if stuck:
            escape_action = random.choice(STUCK_ESCAPE_ACTIONS)
            raw_next_obs, reward, done, info = env.step(escape_action)
            state = frame_stack.step(raw_next_obs)
            episode_reward += reward
            steps += 1
            if info["collision"]:
                collisions += 1
            goal_reached = info["goal_reached"]
            timeout = info["timeout"]
            recent_dists.clear()
            stuck_escapes += 1
            continue

        action = agent.select_action(state)
        raw_next_obs, reward, done, info = env.step(action)
        state = frame_stack.step(raw_next_obs)

        episode_reward += reward
        steps += 1
        recent_dists.append(info["dist_to_goal"])
        if info["collision"]:
            collisions += 1
        goal_reached = info["goal_reached"]
        timeout = info["timeout"]

    return {
        "reward": episode_reward,
        "steps": steps,
        "collisions": collisions,
        "goal_reached": goal_reached,
        "timeout": timeout,
        "stuck_escapes": stuck_escapes,
        "spawn_distance": spawn_distance,
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    scene_path: str,
    episodes: int,
    max_steps: int,
    max_spawn_distance: Optional[float] = None,
) -> dict:
    env_cfg = RoverEnvConfig(
        scene_path=scene_path,
        max_steps=max_steps,
        max_spawn_distance_m=max_spawn_distance,
    )
    env = RoverEnv(config=env_cfg)
    dist_label = "uncapped (full scene)" if max_spawn_distance is None else f"{max_spawn_distance:.1f}m"
    print(f"[evaluate.py] Goal spawn distance: {dist_label}")
    frame_stack = FrameStack(augment=False)

    agent = D3QNAgent(AgentConfig(n_actions=Action.COUNT))
    agent.load(checkpoint_path, eval_mode=True)

    results = []
    try:
        for ep in range(1, episodes + 1):
            result = run_eval_episode(env, frame_stack, agent)
            results.append(result)
            print(
                f"[{ep:4d}/{episodes}] reward={result['reward']:+7.3f} "
                f"steps={result['steps']:4d} "
                f"goal_reached={result['goal_reached']} "
                f"collisions={result['collisions']} "
                f"timeout={result['timeout']} "
                f"stuck_escapes={result['stuck_escapes']}"
            )
    finally:
        env.close()

    success_rate = float(np.mean([r["goal_reached"] for r in results])) * 100
    collision_rate = float(np.mean([r["collisions"] > 0 for r in results])) * 100
    timeout_rate = float(np.mean([r["timeout"] for r in results])) * 100
    avg_reward = float(np.mean([r["reward"] for r in results]))
    success_steps = [r["steps"] for r in results if r["goal_reached"]]
    avg_steps_to_goal = float(np.mean(success_steps)) if success_steps else float("nan")
    avg_stuck_escapes = float(np.mean([r["stuck_escapes"] for r in results]))

    print("=" * 60)
    print(f"Evaluation over {len(results)} episodes -- {checkpoint_path}")
    print("=" * 60)
    print(f"  Success rate (\"accuracy\"):        {success_rate:.1f}%")
    print(f"  Collision rate:                    {collision_rate:.1f}%")
    print(f"  Timeout rate:                       {timeout_rate:.1f}%")
    print(f"  Avg reward:                         {avg_reward:+.3f}")
    print(f"  Avg steps to goal (success only):  {avg_steps_to_goal:.1f}")
    print(f"  Avg stuck-loop escapes/episode:    {avg_stuck_escapes:.2f}")

    return {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "avg_reward": avg_reward,
        "avg_steps_to_goal": avg_steps_to_goal,
        "avg_stuck_escapes": avg_stuck_escapes,
    }


# ---------------------------------------------------------------------------
# Distance sweep
# ---------------------------------------------------------------------------
#
# Split into three reusable pieces so train.py can run a periodic sweep with
# its own live env/agent/frame_stack (no need to rebuild the simulator or
# reload a checkpoint from disk just to sweep), while the CLI path below
# builds its own env/agent from a saved checkpoint.

def bin_distance_sweep(results: list, bin_size_m: float = 1.0) -> tuple:
    """
    Bin run_eval_episode() results by actual spawn distance.

    Returns (bin_centers, bin_success_rates, bin_counts), all lists, with
    empty bins dropped rather than shown as 0%-of-0.
    """
    distances = np.array([r["spawn_distance"] for r in results])
    goal_reached = np.array([r["goal_reached"] for r in results])

    n_bins = max(1, int(np.ceil(distances.max() / bin_size_m)))
    bin_edges = np.arange(0, (n_bins + 1) * bin_size_m, bin_size_m)
    bin_indices = np.clip(np.digitize(distances, bin_edges) - 1, 0, n_bins - 1)

    bin_centers, bin_success_rates, bin_counts = [], [], []
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_centers.append(float(bin_edges[b] + bin_size_m / 2))
        bin_success_rates.append(float(goal_reached[mask].mean()) * 100)
        bin_counts.append(count)

    return bin_centers, bin_success_rates, bin_counts


def plot_distance_sweep(
    bin_centers: list,
    bin_success_rates: list,
    bin_counts: list,
    bin_size_m: float,
    save_path: str,
    title: str,
):
    """Bar chart of success rate per distance bin, annotated with each bin's sample count."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(bin_centers, bin_success_rates, width=bin_size_m * 0.9)
    for c, rate, n in zip(bin_centers, bin_success_rates, bin_counts):
        plt.text(c, rate + 3, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90)
    plt.xlabel("Spawn distance (m)")
    plt.ylabel("Success rate (%)")
    plt.title(title)
    plt.ylim(0, 115)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def evaluate_distance_sweep(
    checkpoint_path: str,
    scene_path: str,
    episodes: int,
    max_steps: int,
    bin_size_m: float = 1.0,
    save_path: Optional[str] = None,
) -> dict:
    """
    Run episodes with goals sampled uncapped across the full scene, bin
    outcomes by actual spawn distance, and report/plot success rate per bin.

    Unlike evaluate(), which reports one success rate at a fixed distance
    cap (matching a curriculum stage), this shows *where* performance falls
    off as goals get farther -- useful for picking the next curriculum
    target or diagnosing a checkpoint's real operating range.
    """
    env_cfg = RoverEnvConfig(scene_path=scene_path, max_steps=max_steps, max_spawn_distance_m=None)
    env = RoverEnv(config=env_cfg)
    print(f"[evaluate.py] Distance sweep: {episodes} episodes, goals uncapped (full scene)")
    frame_stack = FrameStack(augment=False)

    agent = D3QNAgent(AgentConfig(n_actions=Action.COUNT))
    agent.load(checkpoint_path, eval_mode=True)

    results = []
    try:
        for ep in range(1, episodes + 1):
            result = run_eval_episode(env, frame_stack, agent)
            results.append(result)
            print(
                f"[{ep:4d}/{episodes}] spawn_dist={result['spawn_distance']:6.2f}m "
                f"goal_reached={result['goal_reached']} timeout={result['timeout']}"
            )
    finally:
        env.close()

    bin_centers, bin_success_rates, bin_counts = bin_distance_sweep(results, bin_size_m)

    print("=" * 60)
    print(f"Distance sweep over {len(results)} episodes -- {checkpoint_path}")
    print("=" * 60)
    for c, rate, n in zip(bin_centers, bin_success_rates, bin_counts):
        print(f"  {c:5.1f}m: {rate:5.1f}% success (n={n})")

    if save_path:
        plot_distance_sweep(
            bin_centers, bin_success_rates, bin_counts, bin_size_m, save_path,
            title=f"Success Rate vs. Spawn Distance -- {os.path.basename(checkpoint_path)}",
        )
        print(f"\nSaved plot -> {save_path}")

    return {
        "bin_centers": bin_centers,
        "bin_success_rates": bin_success_rates,
        "bin_counts": bin_counts,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained D3QN checkpoint on indoor rover navigation."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--scene", type=str, required=True, help="Path to the .glb scene file.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode before timeout.")
    parser.add_argument(
        "--max-spawn-distance", type=float, default=None,
        help="Cap goal-to-start spawn distance (meters), matching a training curriculum "
             "stage (see checkpoints/<scene>/curriculum_state.json). Default: uncapped, "
             "i.e. goals sampled anywhere on the navmesh -- this may be a harder "
             "distribution than the checkpoint was actually trained on."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--distance-sweep", action="store_true",
        help="Instead of one fixed-distance evaluation, run episodes with goals "
             "sampled uncapped across the full scene and plot success rate as a "
             "function of actual spawn distance. Ignores --max-spawn-distance."
    )
    parser.add_argument(
        "--bin-size", type=float, default=1.0,
        help="Distance bin width in meters for --distance-sweep."
    )
    parser.add_argument(
        "--save-path", type=str, default=None,
        help="Where to save the --distance-sweep plot. Default: "
             "logs/<scene_name>/success_vs_distance.png"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.distance_sweep:
        save_path = args.save_path
        if save_path is None:
            scene_name = os.path.splitext(os.path.basename(args.scene))[0]
            save_path = os.path.join("logs", scene_name, "success_vs_distance.png")
        evaluate_distance_sweep(
            args.checkpoint, args.scene, args.episodes, args.max_steps,
            args.bin_size, save_path,
        )
    else:
        evaluate(args.checkpoint, args.scene, args.episodes, args.max_steps, args.max_spawn_distance)


if __name__ == "__main__":
    main()
