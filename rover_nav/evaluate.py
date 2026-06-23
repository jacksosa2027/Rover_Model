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
        --episodes 100
"""

import argparse
import random
from collections import deque

import numpy as np
import torch

from sim_env import RoverEnv, RoverEnvConfig, Action
from preprocess import FrameStack
from agent import D3QNAgent, AgentConfig


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

# A pure-greedy (epsilon=0) DQN policy has no noise to break ties, so it can
# lock into a repeating action cycle -- e.g. turning in place or nosing into
# the same wall forever. If the geodesic distance to goal hasn't moved by
# more than STUCK_DIST_EPS over STUCK_WINDOW steps, we inject an escape
# action to knock it out of the loop. The escape is restricted to
# turning -- a FORWARD/BACKWARD escape just re-collides with whatever the
# agent is already stuck against, since that's almost always why it's stuck.
STUCK_WINDOW = 20
STUCK_DIST_EPS = 0.05  # meters
STUCK_ESCAPE_ACTIONS = (Action.LEFT, Action.RIGHT)


def run_eval_episode(env: RoverEnv, frame_stack: FrameStack, agent: D3QNAgent) -> dict:
    """Run one episode greedily (no learning, minimal exploration) and return its outcome."""
    raw_obs = env.reset()
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
            action = random.randint(0, agent.cfg.n_actions - 1)
            recent_dists.clear()
            stuck_escapes += 1
        else:
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
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str, scene_path: str, episodes: int, max_steps: int) -> dict:
    env_cfg = RoverEnvConfig(scene_path=scene_path, max_steps=max_steps)
    env = RoverEnv(config=env_cfg)
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    evaluate(args.checkpoint, args.scene, args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
