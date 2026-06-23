"""
Handles of the preprocessing between habitat-sim observation and D3QN input.

Data pipeline:
    - Resize images to (84, 84)
    - Grayscale
    - Normalize to [0, 1]
    - Stack 4 frames -> tensor shape (4, 84, 84)

The 4-frame stack gives the agent temporal context so it can percieve motion from visual input.
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple


FRAME_HEIGHT = 84
FRAME_WIDTH = 84
STACK_SIZE = 4

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert raw RGB(A) frame from habitat-sim into normalized grayscale image ready for stacking.

    Arguments:
        frame:
            numpy array of shape (H, W, 3) or (H, W, 4).
            Habitat-sim returns RGBA by default; the alpha channel (opacity) is stripped automatically.
            
    Returns:
        numpy array of shape (84, 84), dtype float32. Values normalized to [0, 1].
    """

    #Drop alpha channel is present
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    #Resize to standard DQN input size
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

    #Convert to grayscale - reduces dimensionality
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    #Normalize to [0,1]
    frame = frame.astype(np.float32) / 255.0

    return frame

def apply_domain_randomization(frame: np.ndarray) -> np.ndarray:
    """
    Apply lightweight domain randomization to a raw RGB(A) frame before grayscale conversion.
    This helps bridge the sim-to-real gap by making the model robust to the lighting and sensor variations of the pi camera
    module 3.

    Arguments:
        frame: numpy array of shape (H, W, 3) or (H, W, 4), uint8.

    Returns:
        Augmented frame, same shape and dtype.
    """

    #Drop alpha channel for augmentation, restore later if necessary.
    has_alpha = frame.ndim == 3 and frame.shape[2] == 4
    alpha = frame[:, :, 3:] if has_alpha else None
    rgb = frame[:, :, :3]

    #Simulate indoor lighting variation by adjusting brightness and contrast jitter.
    alpha_factor = np.random.uniform(0.7, 1.3) #contrast
    beta_offset = np.random.randint(-30, 30) #brightness
    rgb = cv2.convertScaleAbs(rgb, alpha=alpha_factor, beta=beta_offset)

    #Simulate lens softness/vibration
    if np.random.rand() < 0.3:
        ksize = np.random.choice([3, 5])
        rgb = cv2.GaussianBlur(rgb, (ksize, ksize), sigmaX=0)
    
    #Simulate sensor noise on camera 3 module
    if np.random.rand() < 0.4:
        noise = np.random.normal(0, np.random.uniform(2, 8), rgb.shape).astype(np.int16)
        rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if has_alpha:
        return np.concatenate([rgb, alpha], axis=2)
    return rgb

class FrameStack:
    """
    Maintains a rolling buffer of the last STACK_SIZE preprocessed frames,
    paired with the most recent egocentric goal vector from RoverEnv.

    A raw camera frame can't tell the agent where a randomly-placed goal
    is, so RoverEnv.reset()/step() return (frame, goal_vec) tuples; this
    class stacks the frames for temporal context and passes the goal
    vector through unchanged (it's already a complete instantaneous
    signal, so it isn't stacked).

    Usage:
        stack = FrameStack()
        state = stack.reset((first_frame, goal_vec))  #{"frames": (4,84,84), "goal": (2,)}
        state = stack.step((next_frame, goal_vec))
    """

    def __init__(self, size: int = STACK_SIZE, augment: bool = False):
        """
        Arguments:
            size: number of frames to stack (default 4).
            augment: if True, apply domain randomization before preprocessing. Set True during training, False during
            evaluation and deployment.
        """
        self.size = size
        self.augment = augment
        self._frames: deque = deque(maxlen=size)

    def reset(self, obs: Tuple[np.ndarray, np.ndarray]) -> dict:
        """
        Call at the start of every episode. Fills the buffer with copies of the first frame so there are no 'empty' slots
        at episode start.

        Arguments:
            obs: (frame, goal_vec) tuple from RoverEnv.reset()

        Returns:
            dict with "frames": (STACK_SIZE, 84, 84) float32, "goal": (2,) float32
        """
        frame, goal_vec = obs
        if self.augment:
            frame = apply_domain_randomization(frame)
        processed = preprocess_frame(frame)
        self._frames.clear()
        for _ in range(self.size):
            self._frames.append(processed)
        return self._get_state(goal_vec)

    def step(self, obs: Tuple[np.ndarray, np.ndarray]) -> dict:
        """
        Call every environment step. Adds the new frame and drops the oldest.

        Arguments:
            obs: (frame, goal_vec) tuple from RoverEnv.step()

        Returns:
            dict with "frames": (STACK_SIZE, 84, 84) float32, "goal": (2,) float32
        """
        frame, goal_vec = obs
        if self.augment:
            frame = apply_domain_randomization(frame)
        self._frames.append(preprocess_frame(frame))
        return self._get_state(goal_vec)

    def _get_state(self, goal_vec: np.ndarray) -> dict:
        """
        Stack the buffer into a single array (STACK_SIZE, H, W) and pair
        it with the current goal vector.
        """
        return {
            "frames": np.stack(list(self._frames), axis=0),
            "goal": np.asarray(goal_vec, dtype=np.float32),
        }
