import time
import math
import numpy as np
from roadster_env import RoadsterStuntsEnv
from ppo_agent import PPOAgent

def generate_flip_demonstrations(num_episodes=40):
    """
    Generates expert trajectories for Stunt 1: 180-degree Inverted Drop.
    Executes a fast 0.5s flip maneuver to rotate upright at high altitude,
    cancels lateral drift, and enters a steady retro-burn flare for a soft touchdown.
    """
    env = RoadsterStuntsEnv(mode="flip", dt=0.02)
    env.max_steps = 600

    states = []
    actions = []

    for ep in range(num_episodes):
        env.reset(seed=100 + ep)
        # Drop upside-down from 25m with random tilt perturbations
        env.state = np.array([
            0.0, 0.0, 25.0 + np.random.uniform(-1.0, 1.0),
            0.0, 0.0, -1.0,
            math.pi + np.random.uniform(-0.15, 0.15), 0.0, 0.0,
            0.0, 0.0, 0.0
        ], dtype=np.float32)

        for s in range(env.max_steps):
            r = env.state[6]
            wx = env.state[9]
            z = env.state[2]
            vz = env.state[5]
            vy = env.state[4]

            # Fast flip bang-coast-counterbang logic:
            if abs(r) > 1.2:
                # High torque burst to reach target roll rate of ~6.0 rad/s
                if wx < 6.0:
                    a = np.array([0.95, 0.05, 0.95, 0.05], dtype=np.float32)
                else:
                    a = np.zeros(4, dtype=np.float32)
            elif abs(r) > 0.15 and wx > 0.5:
                # Counter-torque brake right before reaching upright
                a = np.array([0.05, 0.95, 0.05, 0.95], dtype=np.float32)
            else:
                # Retro-burn flare with lateral drift cancellation
                target_roll = -np.clip(vy * 0.04, -math.radians(8.0), math.radians(8.0)) if z > 4.0 else 0.0
                target_v = -2.0 if z > 4.0 else -1.4
                delta_t = 0.35 * (target_v - vz)
                t = np.clip(0.40 + delta_t, 0.0, 1.0)
                roll_err = target_roll - r
                roll_corr = -wx * 0.30 + roll_err * 0.50
                a = np.clip(np.array([t + roll_corr, t - roll_corr, t + roll_corr, t - roll_corr], dtype=np.float32), 0.0, 1.0)

            states.append(env._get_obs())
            actions.append(a)
            obs, rew, term, trunc, info = env.step(a)
            if term or trunc:
                break

    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)


def generate_leap_demonstrations(num_episodes=40):
    """
    Generates expert trajectories for Stunt 2: Tesla Semi Truck Leap.
    Launches forward, hops over the 3.8m high truck, and touches down past the obstacle.
    """
    env = RoadsterStuntsEnv(mode="leap", dt=0.02)
    env.max_steps = 600

    states = []
    actions = []

    for ep in range(num_episodes):
        env.reset(seed=200 + ep)

        for s in range(env.max_steps):
            x = env.state[0]
            z = env.state[2]
            vz = env.state[5]

            # 3-Phase Leap Sequence:
            if s < 18:
                # Boost takeoff to clear 3.8m truck height
                a = np.array([0.85, 0.85, 0.85, 0.85], dtype=np.float32)
            elif x <= 5.0:
                # Coast over truck
                a = np.array([0.30, 0.30, 0.30, 0.30], dtype=np.float32)
            else:
                # Retro-burn flare to touchdown on runway
                target_v = -1.8
                delta_t = 0.30 * (target_v - vz)
                t = np.clip(0.40 + delta_t, 0.0, 1.0)
                a = np.full(4, t, dtype=np.float32)

            states.append(env._get_obs())
            actions.append(a)
            obs, rew, term, trunc, info = env.step(a)
            if term or trunc:
                break

    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)


def train_roadster_agent():
    """
    Trains the PPO neural network agent across both stunts and evaluates its performance.
    """
    print("=========================================================================================")
    print("TRAINING REINFORCEMENT LEARNING AGENT (PPO ACTOR-CRITIC)")
    print("Curriculum: Stunt 1 (180 deg Inverted Drop) + Stunt 2 (Obstacle Leap)")
    print("=========================================================================================\n")

    start_time = time.time()
    agent = PPOAgent(state_dim=15, action_dim=4, lr=0.005)

    # 1) Generate demonstrations
    print("[1/3] Generating expert demonstration rollouts...")
    states_flip, actions_flip = generate_flip_demonstrations(num_episodes=40)
    states_leap, actions_leap = generate_leap_demonstrations(num_episodes=40)
    print(f"      Stunt 1 (Inverted Drop): {len(states_flip)} steps")
    print(f"      Stunt 2 (Semi Leap):     {len(states_leap)} steps")

    # 2) Stage 1 Training: Inverted Drop
    print("\n[2/3] Training policy on Stunt 1 (Inverted Drop)...")
    for epoch in range(1, 151):
        loss = agent.train_imitation_step(states_flip, actions_flip)
        if epoch % 30 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/150 | Inverted Flip Loss: {loss:.5f}")

    # 3) Stage 2 Joint Training: Leap & Inverted Drop
    print("\n[3/3] Joint training on both stunts...")
    all_states = np.vstack([states_flip, states_leap])
    all_actions = np.vstack([actions_flip, actions_leap])

    for epoch in range(1, 121):
        loss = agent.train_imitation_step(all_states, all_actions)
        if epoch % 30 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/120 | Joint Stunts Loss: {loss:.5f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds!")

    # 4) Validation on unseen random seeds
    print("\n-----------------------------------------------------------------------------------------")
    print("RUNNING POST-TRAINING VALIDATION ON UNSEEN SEEDS")
    print("-----------------------------------------------------------------------------------------")

    env_flip = RoadsterStuntsEnv(mode="flip", dt=0.02)
    env_flip.max_steps = 600

    flip_success = 0
    flip_vzs = []
    for t in range(10):
        env_flip.reset(seed=900 + t)
        env_flip.state = np.array([
            0.0, 0.0, 25.0,
            0.0, 0.0, -1.0,
            math.pi + np.random.uniform(-0.15, 0.15), 0.0, 0.0,
            0.0, 0.0, 0.0
        ], dtype=np.float32)

        for s in range(env_flip.max_steps):
            a, _, _ = agent.get_action(env_flip._get_obs(), deterministic=True)
            obs, rew, term, trunc, info = env_flip.step(a)
            if term or trunc:
                if info.get("event") == "successful_landing":
                    flip_success += 1
                flip_vzs.append(env_flip.state[5])
                break

    print(f"Stunt 1 (180 deg Inverted Drop) | Success Rate: {flip_success * 10:5.1f}% | Avg Landing Vz: {np.mean(flip_vzs):5.2f} m/s")

    env_leap = RoadsterStuntsEnv(mode="leap", dt=0.02)
    env_leap.max_steps = 600

    leap_success = 0
    leap_xs = []
    for t in range(10):
        env_leap.reset(seed=950 + t)
        for s in range(env_leap.max_steps):
            a, _, _ = agent.get_action(env_leap._get_obs(), deterministic=True)
            obs, rew, term, trunc, info = env_leap.step(a)
            if term or trunc:
                if info.get("event") == "successful_landing":
                    leap_success += 1
                leap_xs.append(env_leap.state[0])
                break

    print(f"Stunt 2 (Tesla Semi Leap)       | Success Rate: {leap_success * 10:5.1f}% | Avg Touchdown X: {np.mean(leap_xs):5.2f} m")

    # Save trained policy weights
    model_path = "roadster_policy.json"
    agent.save(model_path)
    print(f"\nPolicy weights saved to: '{model_path}'.")
    return agent


if __name__ == "__main__":
    train_roadster_agent()
