"""
agent.py
--------
D3QN (Dueling Double Deep Q-Network) model and agent for indoor rover navigation.

Components:
    D3QN          - Neural network: CNN encoder + dueling value/advantage streams
    ReplayBuffer  - Prioritized experience replay buffer
    D3QNAgent     - Wraps the network with epsilon-greedy policy, learning,
                    and checkpoint save/load

Network input:  (batch, 4, 84, 84)  -- 4-frame grayscale stack from preprocess.py
Network output: (batch, N_ACTIONS)  -- Q-value per discrete action

Double DQN:
    The online network selects the best next action.
    The target network evaluates it.
    This decouples action selection from evaluation, eliminating the
    maximisation bias that causes vanilla DQN to overestimate Q-values.

Dueling streams:
    Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
    Separating state value from action advantage lets the network learn
    which states are inherently good/bad independently of which action
    to take -- critical in corridors where most actions are equivalent.
"""

import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """All D3QN hyperparameters in one place."""

    # --- Network ---
    n_actions: int      = 5         #Must match Action.COUNT in sim_env.py
    input_channels: int = 4         #Frame stack depth (STACK_SIZE in preprocess.py)
    frame_h: int        = 84
    frame_w: int        = 84

    # --- Replay buffer ---
    buffer_size: int    = 100_000   #Max transitions stored
    batch_size: int     = 32        #Transitions sampled per learning step

    # --- Learning ---
    lr: float           = 0.001     #Adam learning rate
    gamma: float        = 0.99      #Discount factor
    grad_clip: float    = 10.0      #Max gradient norm (prevents exploding gradients)

    # --- Exploration (epsilon-greedy) ---
    epsilon_start: float = 1.0      #Start fully random
    epsilon_min: float   = 0.05     #Never drop below 5% random
    epsilon_decay: float = 0.995    #Multiplicative decay per episode

    # --- Target network ---
    target_update_steps: int = 1_000   #Hard-copy online -> target every N steps

    # --- Training warm-up ---
    min_buffer_size: int = 1_000    #Don't start learning until buffer has this many


# ---------------------------------------------------------------------------
#Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-size circular buffer storing (s, a, r, s', done) transitions.

    Uniform random sampling -- straightforward and sufficient for most
    indoor navigation tasks. If you later need faster convergence on
    rare collision events, swap this for a PrioritizedReplayBuffer.
    """

    def __init__(self, capacity: int):
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,   #(4, 84, 84)  float32
        action:     int,
        reward:     float,
        next_state: np.ndarray,   #(4, 84, 84)  float32
        done:       bool,
    ):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Draw a random batch and return stacked numpy arrays.

        Returns:
            states      (B, 4, 84, 84)  float32
            actions     (B,)            int64
            rewards     (B,)            float32
            next_states (B, 4, 84, 84)  float32
            dones       (B,)            float32  (1.0 = terminal)
        """
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# D3QN network
# ---------------------------------------------------------------------------

class D3QN(nn.Module):
    """
    Dueling Double DQN network.

    Architecture:
        Shared CNN encoder
            Conv2d(4 -> 32, k=8, s=4)  -> ReLU   output: (32, 20, 20)
            Conv2d(32-> 64, k=4, s=2)  -> ReLU   output: (64,  9,  9)
            Conv2d(64-> 64, k=3, s=1)  -> ReLU   output: (64,  7,  7)
            Flatten                               output: 3136

        Value stream  V(s)
            Linear(3136 -> 512) -> ReLU
            Linear(512  ->   1)

        Advantage stream  A(s, a)
            Linear(3136 -> 512) -> ReLU
            Linear(512  -> n_actions)

        Q(s, a) = V(s) + A(s, a) - mean_a(A(s, :))

    The mean-subtraction in the advantage stream makes the decomposition
    identifiable -- without it the network can shift value arbitrarily
    between V and A with no change to Q.
    """

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg

        # --- Shared CNN encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        #Compute the encoder's flat output size dynamically
        #so the code stays correct if frame dimensions change.
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.input_channels, cfg.frame_h, cfg.frame_w)
            encoder_out_size = self.encoder(dummy).shape[1]

        # --- Value stream ---
        self.value_stream = nn.Sequential(
            nn.Linear(encoder_out_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

        # --- Advantage stream ---
        self.advantage_stream = nn.Sequential(
            nn.Linear(encoder_out_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg.n_actions),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization for Conv layers, scaled Xavier for Linear.
        Helps stabilize early training compared to PyTorch defaults.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, 84, 84) float32, values in [0, 1].

        Returns:
            Q-values: (B, n_actions) float32.
        """
        features  = self.encoder(x)
        value     = self.value_stream(features)        # (B, 1)
        advantage = self.advantage_stream(features)    # (B, n_actions)

        # Dueling combination with mean-centering
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def get_action(self, x: torch.Tensor) -> int:
        """
        Greedy action for a single state (no gradient).

        Args:
            x: (1, 4, 84, 84) float32 tensor already on the correct device.

        Returns:
            Integer action index.
        """
        with torch.no_grad():
            return int(self.forward(x).argmax(dim=1).item())


# ---------------------------------------------------------------------------
# D3QN agent
# ---------------------------------------------------------------------------

class D3QNAgent:
    """
    Wraps D3QN with:
      - Epsilon-greedy exploration
      - Replay buffer
      - Double DQN learning step
      - Periodic target network sync
      - Checkpoint save / load
    """

    def __init__(self, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()

        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[D3QNAgent] Using device: {self.device}")

        # Two networks: online (trained every step) and target (updated periodically)
        self.online_net = D3QN(self.cfg).to(self.device)
        self.target_net = D3QN(self.cfg).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # Target never produces gradients

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=self.cfg.lr
        )

        self.memory = ReplayBuffer(self.cfg.buffer_size)

        self.epsilon       = self.cfg.epsilon_start
        self.total_steps   = 0
        self.episodes      = 0
        self.losses: deque = deque(maxlen=10_000)

    # -----------------------------------------------------------------------
    # Policy
    # -----------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon: random action (exploration).
        Otherwise:               greedy action from online network (exploitation).

        Args:
            state: numpy array (4, 84, 84), float32, values in [0, 1].

        Returns:
            Integer action index.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.cfg.n_actions - 1)

        state_t = (
            torch.FloatTensor(state)
            .unsqueeze(0)           # (1, 4, 84, 84)
            .to(self.device)
        )
        return self.online_net.get_action(state_t)

    def decay_epsilon(self):
        """
        Multiplicative epsilon decay. Call once per episode end.
        Epsilon never falls below epsilon_min.
        """
        self.epsilon = max(
            self.cfg.epsilon_min,
            self.epsilon * self.cfg.epsilon_decay
        )
        self.episodes += 1

    # -----------------------------------------------------------------------
    # Memory
    # -----------------------------------------------------------------------

    def remember(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """Store a transition in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    # -----------------------------------------------------------------------
    # Learning
    # -----------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """
        Sample a batch from the replay buffer and perform one gradient step.

        Returns the loss value (float) for logging, or None if the buffer
        is not yet large enough to start training.

        Double DQN update rule:
            a* = argmax_a  Q_online(s', a)          (online selects action)
            y  = r + gamma * Q_target(s', a*)       (target evaluates it)
            loss = Huber(Q_online(s, a), y)
        """
        if len(self.memory) < self.cfg.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.cfg.batch_size
        )

        # Move all tensors to device in one block
        states_t      = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # --- Double DQN target ---
        with torch.no_grad():
            # Online net picks the best action in next state
            best_next_actions = self.online_net(next_states_t).argmax(dim=1)  # (B,)
            # Target net evaluates that action
            q_next = self.target_net(next_states_t).gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)                                                        # (B,)
            # Bellman target (zero out terminal states)
            q_target = rewards_t + self.cfg.gamma * q_next * (1.0 - dones_t) # (B,)

        # --- Online net prediction ---
        q_pred = self.online_net(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)                                                            # (B,)

        # Huber loss (less sensitive to outlier rewards than MSE)
        loss = F.smooth_l1_loss(q_pred, q_target)

        # --- Gradient step ---
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.cfg.grad_clip
        )
        self.optimizer.step()

        self.total_steps += 1

        # --- Periodic target network sync ---
        if self.total_steps % self.cfg.target_update_steps == 0:
            self._sync_target()

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def _sync_target(self):
        """Hard copy: online network weights -> target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save(self, path: str):
        """
        Save the full agent state to disk.

        Saves both network weights AND training state (epsilon, step count)
        so training can be resumed exactly where it left off.

        Args:
            path: File path, e.g. "checkpoints/d3qn_ep500.pt"
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "online_net":  self.online_net.state_dict(),
                "target_net":  self.target_net.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
                "epsilon":     self.epsilon,
                "total_steps": self.total_steps,
                "episodes":    self.episodes,
                "config":      self.cfg,
            },
            path,
        )
        print(f"[D3QNAgent] Checkpoint saved -> {path}")

    def load(self, path: str, eval_mode: bool = False):
        """
        Load a checkpoint saved by save().

        Args:
            path:      Path to the .pt checkpoint file.
            eval_mode: If True, set epsilon=0 and put networks in eval mode.
                       Use this for deployment on the Pi or for evaluation runs.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon     = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]
        self.episodes    = checkpoint["episodes"]

        if eval_mode:
            self.epsilon = 0.0
            self.online_net.eval()
            self.target_net.eval()
            print(f"[D3QNAgent] Loaded in eval mode (epsilon=0) <- {path}")
        else:
            self.online_net.train()
            print(
                f"[D3QNAgent] Loaded for continued training "
                f"(epsilon={self.epsilon:.4f}, "
                f"steps={self.total_steps}) <- {path}"
            )

    def export_onnx(self, path: str):
        """
        Export the online network to ONNX for deployment on the Pi 5.
        ONNX Runtime on ARM is ~3x faster than PyTorch for inference.

        Args:
            path: Output file path, e.g. "exports/d3qn_rover.onnx"
        """
        try:
            import onnx  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'onnx' package is required for ONNX export. "
                "Install it with: pip install onnx"
            )

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.online_net.eval()
        dummy = torch.zeros(
            1,
            self.cfg.input_channels,
            self.cfg.frame_h,
            self.cfg.frame_w,
        ).to(self.device)

        torch.onnx.export(
            self.online_net,
            dummy,
            path,
            input_names=["state"],
            output_names=["q_values"],
            dynamic_axes={"state": {0: "batch_size"}},
            opset_version=17,
        )
        print(f"[D3QNAgent] ONNX model exported -> {path}")
        self.online_net.train()

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def recent_loss(self, window: int = 100) -> float:
        """Mean loss over the last `window` learning steps."""
        if not self.losses:
            return 0.0
        return float(np.mean(list(self.losses)[-window:]))

    def __repr__(self):
        buf = len(self.memory)
        return (
            f"D3QNAgent("
            f"device={self.device}, "
            f"epsilon={self.epsilon:.4f}, "
            f"steps={self.total_steps}, "
            f"episodes={self.episodes}, "
            f"buffer={buf}/{self.cfg.buffer_size})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running agent.py smoke test...")

    cfg = AgentConfig(n_actions=5, buffer_size=2_000, min_buffer_size=64)
    agent = D3QNAgent(cfg)
    print(f"  {agent}")

    # Simulate transitions
    for i in range(200):
        state      = np.random.rand(4, 84, 84).astype(np.float32)
        next_state = np.random.rand(4, 84, 84).astype(np.float32)
        action     = agent.select_action(state)
        reward     = np.random.uniform(-1, 1)
        done       = i % 50 == 49

        agent.remember(state, action, reward, next_state, done)
        loss = agent.learn()

        if done:
            agent.decay_epsilon()

    print(f"  After 200 steps: {agent}")
    print(f"  Recent loss: {agent.recent_loss():.6f}")
    assert agent.epsilon < cfg.epsilon_start, "Epsilon did not decay"
    assert len(agent.memory) == 200

    # Save and reload
    agent.save("checkpoints/smoke_test.pt")
    agent2 = D3QNAgent(cfg)
    agent2.load("checkpoints/smoke_test.pt")
    print(f"  Reloaded: {agent2}")

    # ONNX export
    agent.export_onnx("exports/smoke_test.onnx")

    # Verify ONNX output matches PyTorch
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            "exports/smoke_test.onnx",
            providers=["CPUExecutionProvider"]
        )
        dummy = np.random.rand(1, 4, 84, 84).astype(np.float32)
        ort_out = sess.run(None, {"state": dummy})[0]

        agent.online_net.eval()
        with torch.no_grad():
            pt_out = agent.online_net(
                torch.FloatTensor(dummy).to(agent.device)
            ).cpu().numpy()

        max_diff = np.abs(ort_out - pt_out).max()
        print(f"  ONNX vs PyTorch max Q-value diff: {max_diff:.6f}")
        assert max_diff < 1e-4, "ONNX and PyTorch outputs diverged"
        print("  ONNX match OK")
    except ImportError:
        print("  onnxruntime not installed — skipping ONNX verify")

    # Cleanup
    import shutil
    shutil.rmtree("checkpoints", ignore_errors=True)
    shutil.rmtree("exports", ignore_errors=True)

    print("\nAll checks passed.")