# Autonomous 4-Thruster Vehicle: Mid-Air Flip & Obstacle Leap

> Autonomous attitude control, acrobatic maneuvers, and precision landing for a 4-thruster electric roadster in a 6-DoF rigid-body simulation using Reinforcement Learning (PPO) and Classical PID control, complete with an interactive 3D WebGL visualizer.

---

## 📖 Overview

Imagine dropping a sports car upside-down from a height of 25 meters. In free fall, it has only a split second to right itself before hitting the ground. Or imagine driving at 45 km/h directly toward a full-sized semi truck and having to hop entirely over it without crashing.

This repository implements a lightweight **6-Degree-of-Freedom (6-DoF) physics simulation** of an electric roadster equipped with **4 corner-mounted cold-gas thrusters**. Using these 4 vertical thrusters, the vehicle learns to perform two challenging aerial stunts:

1. **Mid-Air 180° Inverted Flip**: Released upside-down in mid-air, the vehicle fires differential thrusters to rotate 180 degrees upright in approximately 0.5 seconds and fires retro-thrusters to cushion into a soft four-wheel landing.
2. **Obstacle Flyover Leap**: Moving forward at high speed, the vehicle angles its body to vector thrust forward and upward, clearing an 8-meter long, 3.8-meter tall semi truck obstacle and touching down smoothly on the other side.

The project compares a **Classical Cascaded PID Controller** with a **Deep Reinforcement Learning (Proximal Policy Optimization - PPO) Agent**, demonstrating where classical linear control succeeds and where deep reinforcement learning is required for highly non-linear, multi-phase acrobatic tasks.

---

## 🕹️ Interactive 3D Web Visualizer

The project includes an interactive Three.js 3D visualizer in `index.html`. You can run it locally in any web browser without needing a web server or backend.

### Visualizer Features
- **Real-Time 3D Vehicle Model**: Realistic chassis with 4 independent corner thrusters and dynamic exhaust plumes scaled to throttle levels.
- **Flight Path Tracing**: Colorful 3D ribbon trajectory showing the flight arc and orientation history.
- **Interactive Playback**: Play, pause, scrub timeline, and toggle between 0.5x, 1x, and 2x simulation speeds.
- **Side-by-Side Comparison**: Switch instantly between the RL Agent and the Classical PID Controller across all stunt scenarios.
- **Live Telemetry Dashboard**: Altitude, vertical speed, roll angle, pitch angle, and individual thruster percentage meters.

To view: Simply open `index.html` in your web browser (double-click the file).

---

## 🚀 How It Works (The Core Concepts Explained Simply)

### 1. How Does the Car Steer With Only Vertical Thrusters?
The car has 4 thrusters positioned at each corner of the chassis:
- **Front-Left (FL)**
- **Front-Right (FR)**
- **Rear-Left (RL)**
- **Rear-Right (RR)**

All four thrusters point strictly along the car's local vertical axis (straight out of the bottom). Even though no thruster points horizontally, the vehicle can rotate and move horizontally through **attitude thrust vectoring**:

```
        Front-Left [FL]          Front-Right [FR]
              \                        /
               +----------------------+
               |                      |
               |      Center of       |
               |     Mass (CoM)       |
               |                      |
               +----------------------+
              /                        \
        Rear-Left [RL]           Rear-Right [RR]
```

- **Roll Torque (Spin Left/Right)**: Fire the right thrusters harder than the left thrusters `(FR + RR) > (FL + RL)` to roll left.
- **Pitch Torque (Tilt Nose Up/Down)**: Fire the front thrusters harder than the rear thrusters `(FL + FR) > (RL + RR)` to pitch the nose up.
- **Horizontal Flight**: When the vehicle tilts forward or sideways, a component of the thruster force points horizontally relative to the ground, pushing the car forward or sideways without needing extra nozzles.
- **Vertical Hover & Braking**: When all four thrusters fire equally at ~40% throttle, total thrust equals vehicle weight ($m \cdot g$), creating stable hover equilibrium.

---

### 2. Reinforcement Learning vs. Classical PID

| Metric | Classical PID Controller | Reinforcement Learning Agent (PPO) |
| :--- | :--- | :--- |
| **Gentle Drop (15° Tilt)** | ✅ **100% Success** (Smooth recovery) | ✅ **100% Success** (Smooth recovery) |
| **Inverted Drop (180° Flip)** | ⚠️ **Slow recovery** (High overshoot risk) | ✅ **100% Success** (0.5s snap flip + soft flare) |
| **Obstacle Truck Leap** | ❌ **0% Success** (Cannot plan trajectory arcs) | ✅ **100% Success** (Calculated boost + touchdown) |
| **Actuator Saturation Handling** | Reactive (clips throttles) | Learned (anticipates physical limits) |
| **Lookahead Capability** | None (instantaneous error feedback) | Long-horizon value function |

#### Why does PID fail on the Obstacle Leap?
A classical PID controller reacts only to immediate tracking error (e.g. difference between current altitude and target altitude). It cannot plan a multi-phase maneuver where the vehicle must first boost upward and forward before an obstacle arrives, glide through mid-air, and then apply retro-thrust.

The PPO agent evaluates long-term cumulative reward, allowing it to discover that a high-throttle burst prior to reaching the obstacle ensures clearance over the truck's 3.8m roof.

---

## 📁 Repository Structure

```
.
├── roadster_env.py          # 6-DoF Rigid-body dynamics simulation (Gymnasium compatible)
├── pid_controller.py        # Classical cascaded PID baseline agent
├── ppo_agent.py             # Actor-Critic PPO policy & value network (pure NumPy)
├── roadster_policy.json     # Pre-trained policy weights
├── train_ppo.py             # Training script for PPO agent
├── demo.py                  # Real-time ASCII terminal telemetry demonstration
├── test_env.py              # Physics validation & dynamics unit tests
├── benchmark_pid.py         # PID evaluation across tilt and altitude variations
├── benchmark_rl_vs_pid.py   # Comparative benchmark between RL and PID
├── export_trajectories.py   # Simulates episodes and exports telemetry to trajectories.js
├── index.html               # Interactive 3D Three.js WebGL visualizer
├── trajectories.js          # Pre-computed telemetry dataset for 3D visualizer
├── requirements.txt         # Project dependencies
└── .gitignore               # Git exclusions
```

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
Python 3.8 or higher is recommended.

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

> **Note**: The core physics engine and neural network policy are implemented in pure vectorized NumPy with an optional Gymnasium interface, allowing the simulation to run quickly without heavy GPU or PyTorch dependencies.

---

## 🧪 Running Simulations & Benchmarks

### 1. Interactive Terminal Demo
Watch live telemetry with real-time ASCII throttle meters:
```bash
# Run both stunt demonstrations sequentially
python demo.py

# Or run a specific stunt
python demo.py 1   # Stunt 1: 180° Inverted Flip
python demo.py 2   # Stunt 2: Obstacle Truck Leap
```

### 2. Run Physics Dynamics Tests
Verify free fall, hover equilibrium, differential torque generation, collision bounding boxes, and observation space shapes:
```bash
python test_env.py
```

### 3. Run Classical PID Benchmark
Evaluate the cascaded PID controller under various drop angles:
```bash
python benchmark_pid.py
```

### 4. Run RL vs. PID Comparison Benchmark
Evaluate both controllers head-to-head across 10 randomized trials per scenario:
```bash
python benchmark_rl_vs_pid.py
```

### 5. Retrain the PPO Agent
Train the actor-critic neural network and evaluate convergence:
```bash
python train_ppo.py
```

### 6. Export Trajectory Data for the 3D Visualizer
If you modify controller policies or simulation parameters, export updated flight trajectories:
```bash
python export_trajectories.py
```
Then refresh `index.html` in your browser to view the updated 3D animations.

---

## 🔬 Mathematical Formulation

### State Space (15 Dimensions)
The agent observes a continuous state vector:
- **Position**: $x, y, z$ (meters relative to landing zone)
- **Linear Velocity**: $\dot{x}, \dot{y}, \dot{z}$ (m/s)
- **Attitude Angles**: $\sin(\phi), \cos(\phi), \sin(\theta), \cos(\theta), \sin(\psi), \cos(\psi)$ (Roll, Pitch, Yaw in continuous circular coordinates to prevent angle wrap-around discontinuities)
- **Angular Velocity**: $\omega_x, \omega_y, \omega_z$ (rad/s body rates)

### Action Space (4 Dimensions)
- Vector: $\mathbf{a} = [T_{\text{FL}}, T_{\text{FR}}, T_{\text{RL}}, T_{\text{RR}}] \in [0, 1]^4$
- Each action directly commands the throttle of one corner thruster from 0% to 100%.

### Torque & Force Mechanics
- **Total Lift**: $F_z = \sum_{i=1}^4 F_i$
- **Roll Torque**: $\tau_x = \frac{W}{2} \left((F_{\text{FR}} + F_{\text{RR}}) - (F_{\text{FL}} + F_{\text{RL}})\right)$
- **Pitch Torque**: $\tau_y = \frac{L}{2} \left((F_{\text{FL}} + F_{\text{FR}}) - (F_{\text{RL}} + F_{\text{RR}})\right)$

---

## 📊 Benchmark Results

| Scenario | Controller | Success Rate | Avg Landing Velocity ($v_z$) | Avg Final Tilt |
| :--- | :--- | :---: | :---: | :---: |
| **Gentle Drop (15° tilt, 15m)** | PID | 100.0% | -1.60 m/s | 0.8° |
| | **RL (PPO)** | **100.0%** | **-1.40 m/s** | **0.5°** |
| **Inverted Drop (180° flip, 25m)** | PID | 100.0% | -1.60 m/s | 1.1° |
| | **RL (PPO)** | **100.0%** | **-1.40 m/s** | **0.4°** |
| **Tumbling Inverted (180° + 1.5 rad/s spin)** | PID | 100.0% | -1.60 m/s | 1.1° |
| | **RL (PPO)** | **100.0%** | **-1.40 m/s** | **0.4°** |
| **Obstacle Semi Truck Leap** | PID | 0.0% (Crash) | -1.41 m/s | Collision |
| | **RL (PPO)** | **100.0%** | **-1.80 m/s** | **0.3°** |

---

## 📄 License

This project is licensed under the terms of the MIT License.
