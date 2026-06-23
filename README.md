# Dueling Double Deep Q-Network (D3QN) CV Model

This codebase is used to generate computer vision models that will be used for the autonomous navigation of both a rover and drone.

# Table of Contents

- [D3QN CV Model](#dueling-double-deep-q-network-d3qn-cv-model)
- [Background](#background)
- [Install](#install)
- [usage](#usage)
- [Using your own scans](#using-your-own-scans)
- [Evaluating a trained model](#evaluating-a-trained-model)
- [License](#license)

# Background

A Dueling Double Deep Q-Network (D3QN) is a deep reinforcement learning (DRL) algorithm that combines two critical advancements in the DQN family: the dueling network architecture and double Q-learning. This hybrid architecture has been empirically validated to improve the efficiency and stability of value-based deep reinforcement learning in high-dimensional, noisy environments across robotics.

D3QN unifies two enhancements to the original DQN:

- **Dueling Architecture:** The value function $V(s)$ and the advantage function $A(s,a)$ are estimated in parallel, with the final Q-value computed as

$$Q(s,a;\theta) = V(s;\theta,\beta) + \left[A(s,a;\theta,\alpha) - \frac{1}{|A|}\sum_{a'} A(s,a';\theta,\alpha)\right].$$

  This formulation enables the network to learn the state-value function independently of the action, improving evaluation in settings with many similar-valued actions.

- **Double Q-Learning:** To address maximization bias, D3QN decouples the action selection and evaluation in the target:

$$y_t = r_t + \gamma Q(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a'; \theta_t); \theta^-).$$

# Install

## Prerequisites

- Linux x86_64 with an NVIDIA GPU (the pinned PyTorch/Habitat-Sim builds expect CUDA 12.1-compatible drivers)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)

## 1. Clone the repo

```bash
git clone https://github.com/jacksosa2027/sam_wheeled_model.git
cd sam_wheeled_model
```

## 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate wheeled_model
```

This pulls in Habitat-Sim 0.3.3 (bullet build) from the `aihabitat` channel along with PyTorch/CUDA 12.1 and the rest of the pinned dependencies — no source build is required to train.

## 3. Download a scene dataset

Training and evaluation need at least one `.glb` scene to navigate. The Habitat test scenes are enough to get started:

```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

For larger/more realistic environments (HM3D, MP3D, ReplicaCAD, etc.), see [habitat-sim's DATASETS.md](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md) — some of these require requesting access from their providers before they can be downloaded.

## 4. (Optional) Clone habitat-sim from source

`rover_nav/train.py` falls back to `../habitat-sim/data/scene_datasets/habitat-test-scenes/apartment_1.glb` if you don't pass `--scene`, and `how_to_run_sim.txt` uses `habitat-sim/examples/viewer.py` to preview scenes. Neither is required to train, but if you want them, clone the matching version alongside this repo:

```bash
cd ..
git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-sim.git
```

# Usage

Train a D3QN agent from scratch:

```bash
python rover_nav/train.py \
  --scene data/scene_datasets/habitat-test-scenes/apartment_1.glb \
  --episodes 5000
```

Resume training from a saved checkpoint:

```bash
python rover_nav/train.py \
  --scene data/scene_datasets/habitat-test-scenes/apartment_1.glb \
  --resume checkpoints/<run_name>/d3qn_latest.pt
```

Checkpoints are written to `checkpoints/<run_name>/` and training progress to `logs/`. Run `python rover_nav/train.py --help` for the full list of options (curriculum, domain randomization, max steps per episode, seed, etc.).

# Using your own scans

You aren't limited to the downloaded scene datasets — you can train and evaluate on a scan of your own space. Scan a room or apartment with a 3D capture app such as [Polycam](https://poly.cam/) (LiDAR or photo mode both work) and export it as a `.glb` mesh.

Point `--scene` at the exported file:

```bash
python rover_nav/train.py --scene /path/to/my_scan.glb --episodes 5000
```

The first time a scene is loaded, `RoverEnv` checks for a baked-in navmesh; if the `.glb` doesn't have one (which is normal for a Polycam export), it automatically recomputes one with Habitat-Sim's default `NavMeshSettings` and saves it alongside the scene as `my_scan.navmesh`. Subsequent runs against the same `.glb` reuse that cached navmesh instead of recomputing it.

# Evaluating a trained model

`rover_nav/evaluate.py` runs a trained checkpoint greedily (epsilon=0, no exploration) for a number of episodes and reports aggregate metrics — success rate, collision rate, timeout rate, average reward, and average steps-to-goal:

```bash
python rover_nav/evaluate.py \
  --checkpoint checkpoints/<run_name>/d3qn_best.pt \
  --scene data/scene_datasets/habitat-test-scenes/apartment_1.glb \
  --episodes 100
```

`--scene` can point at any `.glb`, including your own scans (see [Using your own scans](#using-your-own-scans)). Run `python rover_nav/evaluate.py --help` for the full list of options (`--max-steps`, `--seed`, etc.).

# License

ADD LATER IF NECESSARY
