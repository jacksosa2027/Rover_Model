"""
Habitat-sim environment wrapper

Wraps habitat-sim into a clean gym-style interface:
    env = RoverEnv(scene_path=path/to/scene.glb)
    state = env.reset()
    state, reward, done, info = env.step(action)

Action space (discrete):
    0 - Move forward
    1 - Move backward
    2 - Turn left
    3 - Turn right
    4 - Stop (no-op)

Observation:
    Raw RGB(A) numpy array (480, 640, 4) - passed to FrameStack in train.py.

Coordinate note:
    Habitat-sim uses a Y-up, right-handed coordinate system.
    All position/rotation values are in metres and radians respectively.
"""

import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import quaternion

try:
    import habitat_sim
    import habitat_sim.agent
except ImportError as e:
    raise ImportError(
        "habitat-sim is not installed or not portable. \n"
        "Install it from: https://github.com/facebookresearch/habitat-sim\n"
        f"Original error: {e}"
    )

#Config dataclass ----------------------------------------------------
@dataclass
class RoverEnvConfig:
    """
    All tunable parameters for the simulation environment.
    Edit these values rather than touching the class logic below.
    """

    #Scene
    scene_path: str = "" #Path to a .glb scene file

    #Camera (Matched to Pi camera module 3)
    resolution_h: int = 480         #Render height in pixels
    resolution_w: int = 640         #Render width in pixels
    hfov_degree: float = 66.0       #Horizontal field of view
    camera_height_m: float = 0.15   #Camera height above rover base (meters)

    #Agent movement
    move_step_m: float = 0.25       #Distance per foward/backward action (meters)
    turn_step_degree: float = 10.0  #Degrees per left/right turn action

    #Episode
    max_steps: int = 500                #Hard cut-off per episode
    goal_radius_m: float = 0.5          #Distance at which goal is considered reached
    collision_penalty: float = -1.0     #Reward on collision
    goal_reward: float = 10.0           #Reward on reaching goal
    progress_scale: float = 5.0         #Multiplier on geodesic progress reward
    time_penalty: float = 0.01          #Per-step penalty to encourage efficiency

    #Spawn randomization
    min_spawn_distance_m: float = 1.5   #Min gaol-to-start distance at episode start
    max_spawn_attempts: int = 50        #Attempts to find a valid random spawn

    #Sensor noise (optional, adds realism)
    add_sensor_noise: bool = False

#Action definitions ------------------------------------------------------------ 
class Action:
    FORWARD  = 0
    BACKWARD = 1
    LEFT     = 2
    RIGHT    = 3
    STOP     = 4
    COUNT    = 5

#Main environment class --------------------------------------------------------
class RoverEnv:
    """
    Gym-style wrapper around habitat-sim for indoor rover navigation

    The rover observes the world only though its forward facing RGB camera,
    matching the single sensor of the Pi rover.
    """

    def __init__(self, config: Optional[RoverEnvConfig] = None, scene_path: str = ""):
        """
        Arguments:
            config: RoverEnvConfig instance. If None, defaults are used.
            scene_path: Conveniece override for config.scene_path
        """

        self.cfg = config or RoverEnvConfig()
        if scene_path:
            self.cfg.scene_path = scene_path
        
        if not self.cfg.scene_path:
            raise ValueError(
                "scene_path must be provided via RoverEnvConfig or "
                "the scene_path argument. \n"
                "Download test scenes with: \n"
                "  python -m habitat_sim.utils.datasets_download "
                "--uris habitat_test_scenes --data-path data/"
            )

        self._sim: Optional[habitat_sim.Simulator] = None
        self._agent: Optional[habitat_sim.agent.Agent] = None

        self._step_count: int = 0
        self._prev_dist: float = 0
        self._goal_pos: Optional[np.ndarray] = None
        self._episode_count: int = 0

        self._build_simulator()

    #Public interface ---------------------------------------------------
    def reset(self) -> np.ndarray:
        """
        Start a new episode.
        Spawns the agent and goal at random navigable positions that are
        at least min_spawn_distance_m apart.

        Returns:
            Raw RGB(A) observation, shape (H, W, 4), dtype uint8.
        """

        self._episode_count += 1
        self._step_count = 0

        #Randomize the scene's lighting every N episodes for domain randomization
        if self._episode_count % 10 == 0:
            self._randomize_lighting()

        #Place agent and goal
        self._goal_pos = self._sample_navigable_point()
        agent_pos = self._sample_navigable_point_near(
            avoid=self._goal_pos,
            min_dist=self.cfg.min_spawn_distance_m
        )
        self._place_agent(agent_pos)

        #init distance to goal for progress reward
        self._prev_dist = self._geodesic_distance(
            self._get_agent_pos(),
            self._goal_pos
        )

        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one action in the environment.

        Arguments:
            action: Integer from Action.FORWARD,...,Action.STOP.

        Returns:
            obs:    RGB(A) numpy array (H, W, 4)
            reward: float
            done:   True if episode ended (goal reached, collision, timeout)
            info:   dict with diagnostic fields
        """

        assert 0 <= action < Action.COUNT, f"Invalid action: {action}"

        self._step_count += 1
        collision = self._execute_action(action)
        obs = self._get_observation()

        agent_pos = self._get_agent_pos()
        curr_dist = self._geodesic_distance(agent_pos, self._goal_pos)
        goal_reached = curr_dist < self.cfg.goal_radius_m
        timeout = self._step_count >= self.cfg.max_steps

        reward = self.compute_reward(
            prev_dist = self._prev_dist,
            curr_dist = curr_dist,
            collision = collision,
            goal_reached = goal_reached
        )
        self._prev_dist = curr_dist
        done = goal_reached or timeout

        info = {
            "step": self._step_count,
            "dist_to_goal": curr_dist,
            "collision": collision,
            "goal_reached": goal_reached,
            "timeout": timeout,
            "episode": self._episode_count,
        }

        return obs, reward, done, info
    
    def close(self):
        """
        Release the habitat-sim and free GPU resources
        """
        if self._sim is not None:
            self._sim.close()
            self._sim = None
    
    def get_action_count(self) -> int:
        return Action.COUNT
    
    def get_observation_shape(self) -> Tuple[int, int, int]:
        """
        Returns (H, W, 4) - The raw RGB(A) frame shape
        """
        return (self.cfg.resolution_h, self.cfg.resolution_w, 4)
    

    #Simulator construction ---------------------------------------------
    def _build_simulator(self):
        """
        Construct and initialize the habitat-sim simulator.
        """

        #Backend config
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.cfg.scene_path
        backend_cfg.enable_physics = True #Necessary for collision detection

        #RGB camera sensor
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "rgb_camera"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [self.cfg.resolution_h, self.cfg.resolution_w]
        rgb_sensor.hfov = self.cfg.hfov_degree
        rgb_sensor.position = [0.0, self.cfg.camera_height_m, 0.0]
        rgb_sensor.orientation = [0.0, 0.0, 0.0] #Level, forward-facing

        #Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor]

        #Register the 5 discrete actions
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=self.cfg.move_step_m)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward",
                habitat_sim.agent.ActuationSpec(amount=self.cfg.move_step_m)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=self.cfg.turn_step_degree)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=self.cfg.turn_step_degree)
            ),
            "stop": habitat_sim.agent.ActionSpec(
                "stop",
                habitat_sim.agent.ActuationSpec(amount=0)
            ),
        }

        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(sim_cfg)
        self._agent = self._sim.initialize_agent(0)

    ACTION_NAMES = {
        Action.FORWARD: "move_forward",
        Action.BACKWARD: "move_backward",
        Action.LEFT: "turn_left",
        Action.RIGHT: "turn_right",
        Action.STOP: "stop",
    }

    def _execute_action(self, action: int) -> bool:
        """
        Execute the action and return whether a collision occured

        Habitat-sim reports collisions via agent state after movement
        """
        action_name = self.ACTION_NAMES[action]
        self._agent.act(action_name)

        #Check for collision (only meaningful for movement actions)
        if action in (Action.FORWARD, Action.BACKWARD):
            agent_state = self._agent.get_state()
            return bool(
                self._sim.previous_step_collided
                if hasattr(self._sim, "previous_step_collided")
                else False
            )
        return False
    
    #Reward -----------------------------------------------------
    def compute_reward(
            self,
            prev_dist: float,
            curr_dist: float,
            collision: bool,
            goal_reached: bool,
    ) -> float:
        """
        Shaped reward combining:
            - Geodesic progress towards goal (dense signal)
            - Colision pentalty (safety)
            - Goal bonus (sparse terminal signal)
            Time penalty (efficiency)
        """
        reward = 0.0

        #Dense: reward progress, penalize moving away from goal
        reward += (prev_dist - curr_dist) * self.cfg.progress_scale

        #Collision penalty
        if collision:
            reward += self.cfg.collision_penalty

        #Terminal: goal reached
        if goal_reached:
            reward += self.cfg.goal_reward

        #Per-step time penalty
        reward -= self.cfg.time_penalty
        
        return float(reward)
    
    #Navigation helpers
    def _get_agent_pos(self) -> np.ndarray:
        """
        Returns the agent's XYZ position in world coordinates
        """
        return np.array(self._agent.get_state().position)
    
    def _geodesic_distance(self, from_pos: np.ndarray, to_pos: np.ndarray) -> float:
        """
        compute the navigable (geodesic) path distance between two points.
        Falls back to Euclidean distance if no pathfinder exists.
        """
        path = habitat_sim.ShortestPath()
        path.requested_start = from_pos.tolist()
        path.requested_end = to_pos.tolist()
        found = self._sim.pathfinder.find_path(path)
        if found and path.geodesic_distance < float("inf"):
            return path.geodesic_distance
        #Fallback: straight-line distance
        return float(np.linalg.norm(from_pos - to_pos))
    
    def _sample_navigable_point(self) -> np.ndarray:
        """
        Return a random navigable position in the scene.
        """
        return np.array(self._sim.pathfinder.get_random_navigable_point())
    
    def _sample_navigable_point_near(
        self,
        avoid: np.ndarray,
        min_dist: float,
    ) -> np.ndarray:
        """
        Sample a navigable point that is at least min_dist meters from the avoid
        position. Falls back to any navigable point if no valid spawn is found 
        within max_psawn_attempts tries
        """
        for _ in range(self.cfg.max_spawn_attempts):
            candidate = self._sample_navigable_point()
            dist = self._geodesic_distance(candidate, avoid)
            if dist >= min_dist:
                return candidate
        
        #Fallback: any navigable point (episode will be easy but valid)
        return self._sample_navigable_point()
    
    def _place_agent(self, position: np.ndarray):
        """
        Teleport the agent to position with a random yaw rotation.
        """
        state = habitat_sim.agent.AgentState()
        state.position = position.tolist()

        #Random initial heading so that the agent doesn't always face the same way
        yaw_rad = random.uniform(0, 2 * math.pi)
        #Habitat-sim uses quaternion (x, y, z, w) for rotation.
        state.rotation = np.quaternion(
            math.cos(yaw_rad / 2), #W
            0.0,                   #x
            math.sin(yaw_rad / 2), #y-axis yaw
            0.0                    #z
        )

        assert self._agent is not None
        self._agent.set_state(state)

    #Observation ----------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """
        Render the current frame and return it as a (H, W, 4) RGB(A) array.
        Optionally adds Guassian noise to simulate camera sensor noise.
        """
        obs = self._sim.get_sensor_observations()
        frame = obs["rgb_camera"] #shape (H, W, 4), dtype unit8

        if self.cfg.add_sensor_noise:
            noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    #Domain randomization ----------------------------------------------------
    def _randomize_lighting(self):
        """
        Randomize the scene's ambient and directional lighting intensity.
        Called automatically every 10 episodes to expose agent to different
        indoor lighting conditions during training.

        This requires a habitat-sim build with lighting support
        Silently skips if lighting control is unavailable
        """

        try:
            light_setup = []
            n_lights = random.randint(1, 3)
            for i in range(n_lights):
                light = habitat_sim.LightInfo(
                    vector=[
                        random.uniform(-1, 1),
                        random.uniform(0.5, 1.5),
                        random.uniform(-1, 1),
                        0.0,
                    ],
                    color=[
                        random.uniform(0.7, 1.0),
                        random.uniform(0.7, 1.0),
                        random.uniform(0.6, 0.9),
                    ],
                    model=habitat_sim.LightPositionModel.Global,
                )
                light_setup.append(light)
            self._sim.set_light_setup(light_setup)
        except (AttributeError, Exception):
            #Lighting API not available in this build
            print("Lighitng API not available in this build. Skipping.")
            pass

    #Context manager support ------------------------------------------------
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return (
            f"RoverEnv(scene='{self.cfg.scene_path}', "
            f"steps={self._step_count}, episode={self._episode_count})"
        )
    
    #Smoke test -----------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # Look for a test scene in the default habitat-sim data directory
    DEFAULT_SCENE = os.path.join(
        os.path.dirname(__file__),
        "..", "habitat-sim", "data",
        "scene_datasets", "habitat-test-scenes", "apartment_1.glb"
    )

    scene = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE

    if not os.path.exists(scene):
        print(
            f"Scene not found at: {scene}\n"
            "Download test scenes with:\n"
            "  python -m habitat_sim.utils.datasets_download "
            "--uris habitat_test_scenes --data-path ../habitat-sim/data/\n"
            "Or pass a custom scene path as an argument:\n"
            "  python sim_env.py path/to/scene.glb"
        )
        sys.exit(1)

    print(f"Loading scene: {scene}")
    cfg = RoverEnvConfig(scene_path=scene, max_steps=20)

    with RoverEnv(config=cfg) as env:
        print(f"  Action count:       {env.get_action_count()}")
        print(f"  Observation shape:  {env.get_observation_shape()}")

        obs = env.reset()
        print(f"  reset() -> obs.shape={obs.shape}, dtype={obs.dtype}")
        assert obs.shape == (480, 640, 4)

        total_reward = 0.0
        for step in range(20):
            action = random.randint(0, Action.COUNT - 1)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print(
                f"  step {step+1:02d} | action={action} | "
                f"reward={reward:+.3f} | dist={info['dist_to_goal']:.2f}m | "
                f"collision={info['collision']} | done={done}"
            )
            if done:
                print("  Episode ended early.")
                break

        print(f"\n  Total reward: {total_reward:.3f}")
        print("Smoke test passed.")








