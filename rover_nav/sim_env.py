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

import os
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
    max_spawn_distance_m: Optional[float] = None  #Max goal-to-start distance; None = uncapped.
                                                   #Set via RoverEnv.set_max_spawn_distance() to
                                                   #drive a training curriculum (see train.py).
    max_spawn_attempts: int = 50        #Attempts to find a valid random spawn

    #Goal vector normalization
    goal_dist_norm_m: float = 10.0      #Distance (m) that normalizes to 1.0 in the goal vector

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
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Start a new episode.
        Spawns the agent and goal at random navigable positions that are
        at least min_spawn_distance_m apart.

        Returns:
            frame:    Raw RGB(A) observation, shape (H, W, 4), dtype uint8.
            goal_vec: Egocentric goal vector, shape (2,) float32 --
                      see _goal_vector() for details. A vision-only frame
                      can't disambiguate where a randomly-placed goal is,
                      so this is returned alongside it.
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
            min_dist=self.cfg.min_spawn_distance_m,
            max_dist=self.cfg.max_spawn_distance_m,
        )
        self._place_agent(agent_pos)

        #init distance to goal for progress reward
        self._prev_dist = self._geodesic_distance(
            self._get_agent_pos(),
            self._goal_pos
        )

        goal_vec = self._goal_vector(self._get_agent_pos(), self._prev_dist)
        return self._get_observation(), goal_vec

    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, dict]:
        """
        Execute one action in the environment.

        Arguments:
            action: Integer from Action.FORWARD,...,Action.STOP.

        Returns:
            obs:    (frame, goal_vec) tuple -- see reset() for details.
            reward: float
            done:   True if episode ended (goal reached, collision, timeout)
            info:   dict with diagnostic fields
        """

        assert 0 <= action < Action.COUNT, f"Invalid action: {action}"

        self._step_count += 1
        collision = self._execute_action(action)
        frame = self._get_observation()

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

        goal_vec = self._goal_vector(agent_pos, curr_dist)

        info = {
            "step": self._step_count,
            "dist_to_goal": curr_dist,
            "collision": collision,
            "goal_reached": goal_reached,
            "timeout": timeout,
            "episode": self._episode_count,
        }

        return (frame, goal_vec), reward, done, info
    
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

    def get_goal_dim(self) -> int:
        """
        Returns the length of the goal vector returned alongside each frame.
        Must match AgentConfig.goal_dim in agent.py.
        """
        return 2

    def set_max_spawn_distance(self, max_dist: Optional[float]):
        """
        Update the max goal-to-start spawn distance used by future reset()
        calls, without rebuilding the simulator. Pass None to remove the cap.

        Used by train.py to drive a curriculum that starts with nearby goals
        and expands the spawn radius as the agent's success rate improves.
        """
        self.cfg.max_spawn_distance_m = max_dist

    def estimate_max_geodesic_distance_m(self, samples: int = 200, percentile: float = 95.0) -> float:
        """
        Estimate the hardest-case goal distance in this scene by sampling
        random navigable point pairs and measuring their actual geodesic
        distance -- the same distance metric used for rewards and for
        max_spawn_distance_m, not a geometric proxy.

        The navmesh's bounding-box diagonal was tried first and rejected: it
        only bounds straight-line separation between two points, but
        geodesic paths have to go around walls, so two points sitting close
        together inside a modest bounding box can still be a long walk
        apart in a winding corridor. Sampling real paths is the only way to
        size a distance curriculum so its final stage actually covers the
        hardest cases in *this* scene, instead of an arbitrary fixed
        ceiling that may be far smaller than the real building.
        """
        distances = []
        for _ in range(samples):
            a = self._sample_navigable_point()
            b = self._sample_navigable_point()
            dist = self._geodesic_distance(a, b)
            if dist < float("inf"):
                distances.append(dist)

        if not distances:
            return 0.0
        return float(np.percentile(distances, percentile))


    #Simulator construction ---------------------------------------------
    def _build_simulator(self):
        """
        Construct and initialize the habitat-sim simulator.
        """

        #Backend config
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.cfg.scene_path
        backend_cfg.enable_physics = False
        backend_cfg.gpu_device_id = 0

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
        }

        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(sim_cfg)
        self._agent = self._sim.initialize_agent(0)

        if not self._sim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings)
            navmesh_path = os.path.splitext(self.cfg.scene_path)[0] + ".navmesh"
            self._sim.pathfinder.save_nav_mesh(navmesh_path)
            print(f"[RoverEnv] Navmesh saved to {navmesh_path}")

    ACTION_NAMES = {
        Action.FORWARD: "move_forward",
        Action.BACKWARD: "move_backward",
        Action.LEFT: "turn_left",
        Action.RIGHT: "turn_right",
        Action.STOP: "stop",
    }

    def _execute_action(self, action: int) -> bool:
        """
        Execute the action and return whether a collision occured.

        habitat_sim.agent.Agent.act() returns this directly -- there is no
        separate `previous_step_collided` flag on the simulator.
        """
        if action == Action.STOP:
            return False

        action_name = self.ACTION_NAMES[action]
        return bool(self._agent.act(action_name))
    
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
    def _goal_vector(self, agent_pos: np.ndarray, dist: float) -> np.ndarray:
        """
        Egocentric goal vector: [normalized_distance, normalized_relative_heading].

        A raw camera frame can't tell the agent which way a randomly-placed
        goal is, so this gets passed through the pipeline alongside the
        frame (see preprocess.FrameStack and agent.D3QN) as a PointGoal-style
        sensor would in habitat-lab.

        normalized_distance:  geodesic distance / goal_dist_norm_m, clipped to [0, 1]
        normalized_heading:   angle from the agent's facing direction to the
                               goal, in [-1, 1] where 0 = straight ahead,
                               +/-1 = directly behind.
        """
        agent_state = self._agent.get_state()
        q = agent_state.rotation
        yaw = 2.0 * math.atan2(q.y, q.w)

        dx = self._goal_pos[0] - agent_pos[0]
        dz = self._goal_pos[2] - agent_pos[2]
        bearing = math.atan2(-dx, -dz)

        rel_heading = bearing - yaw
        rel_heading = math.atan2(math.sin(rel_heading), math.cos(rel_heading))

        norm_dist = min(dist / self.cfg.goal_dist_norm_m, 1.0)
        norm_heading = rel_heading / math.pi

        return np.array([norm_dist, norm_heading], dtype=np.float32)

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
        max_dist: Optional[float] = None,
    ) -> np.ndarray:
        """
        Sample a navigable point whose geodesic distance from `avoid` is at
        least min_dist meters and, if max_dist is given, at most max_dist
        meters. The upper bound lets train.py run a curriculum that starts
        with nearby goals and expands the spawn radius as the agent improves,
        instead of sampling goals anywhere on the navmesh from episode 1.

        Falls back to the closest min_dist-satisfying candidate seen (or any
        navigable point, if none satisfied even that) if no candidate
        satisfies both bounds within max_spawn_attempts tries.
        """
        fallback = None
        for _ in range(self.cfg.max_spawn_attempts):
            candidate = self._sample_navigable_point()
            dist = self._geodesic_distance(candidate, avoid)
            if dist < min_dist:
                continue
            if max_dist is None or dist <= max_dist:
                return candidate
            if fallback is None:
                fallback = candidate

        return fallback if fallback is not None else self._sample_navigable_point()
    
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
        print(f"  Goal vector dim:    {env.get_goal_dim()}")

        frame, goal_vec = env.reset()
        print(f"  reset() -> frame.shape={frame.shape}, dtype={frame.dtype}, goal_vec={goal_vec}")
        assert frame.shape == (480, 640, 4)
        assert goal_vec.shape == (2,)

        total_reward = 0.0
        for step in range(20):
            action = random.randint(0, Action.COUNT - 1)
            (frame, goal_vec), reward, done, info = env.step(action)
            total_reward += reward
            print(
                f"  step {step+1:02d} | action={action} | "
                f"reward={reward:+.3f} | dist={info['dist_to_goal']:.2f}m | "
                f"goal_vec={goal_vec} | "
                f"collision={info['collision']} | done={done}"
            )
            if done:
                print("  Episode ended early.")
                break

        print(f"\n  Total reward: {total_reward:.3f}")
        print("Smoke test passed.")








