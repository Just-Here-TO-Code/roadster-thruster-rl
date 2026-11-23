"""
Interactive Terminal Demonstration of 4-Thruster Vehicle Stunts.
Demonstrates:
  1) Mid-Air 180-Degree Inverted Flip & Safe Landing Cushion
  2) High-Speed Obstacle Leap over a Semi Truck
"""

import sys
import time
import math
import numpy as np
from roadster_env import RoadsterStuntsEnv

def make_bar(value, max_val=1.0, length=12):
    """
    Creates an ASCII progress bar like [######      ] for throttles.
    """
    ratio = max(0.0, min(1.0, value / max_val))
    filled = int(round(ratio * length))
    return "[" + "#" * filled + " " * (length - filled) + "]"

def run_stunt1_demo():
    """
    Demonstrates Stunt 1: Upside-Down Drop (The 180-degree Mid-Air Flip).
    Uses differential thrust to induce fast rotation upright and cushion the touchdown.
    """
    print("\n=======================================================")
    print("DEMO: STUNT 1 - UPSIDE-DOWN DROP (MID-AIR FLIP)")
    print("Vehicle released at 25m altitude upside-down (roll = 180 deg).")
    print("Watch corner thrusters fire differentially to flip upright!")
    print("=======================================================\n")

    env = RoadsterStuntsEnv(mode="flip", dt=0.04) # 25 FPS display
    obs, info = env.reset(seed=101)

    # Initial upside-down state
    env.state = np.array([0.0, 0.0, 25.0, 0.0, 0.0, -1.0, math.pi, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    total_reward = 0.0
    for step in range(120):
        x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = env.state

        action = np.zeros(4, dtype=np.float32)

        if abs(roll) > math.radians(20.0):
            # Mid-air flip: burst left thrusters to rotate clockwise
            if wx < 3.0:
                action = np.array([0.9, 0.0, 0.9, 0.0], dtype=np.float32) # Left burst
            else:
                action = np.array([0.1, 0.5, 0.1, 0.5], dtype=np.float32) # Counter-torque brake
            phase_text = "FLIPPING 180 DEG"
        else:
            # Leveling and retro-cushion
            base_throttle = 0.40 # Hover equilibrium
            descent_brake = max(0.0, (-vz - 1.5) * 0.15)
            level_roll = -wx * 0.2
            level_pitch = -wy * 0.2

            t_fl = base_throttle + descent_brake + level_roll + level_pitch
            t_fr = base_throttle + descent_brake - level_roll + level_pitch
            t_rl = base_throttle + descent_brake + level_roll - level_pitch
            t_rr = base_throttle + descent_brake - level_roll - level_pitch

            action = np.clip(np.array([t_fl, t_fr, t_rl, t_rr], dtype=np.float32), 0.0, 1.0)
            phase_text = "CUSHIONING DESCENT"

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 3 == 0 or terminated:
            t_sec = step * env.dt
            alt_bar = make_bar(env.state[2], max_val=25.0, length=10)
            fl_bar = make_bar(action[0], length=6)
            fr_bar = make_bar(action[1], length=6)

            print(f"t={t_sec:4.2f}s | Alt:{env.state[2]:5.1f}m {alt_bar} | Vz:{env.state[5]:5.1f}m/s | "
                  f"Roll:{math.degrees(env.state[6]):6.1f} deg | FL:{fl_bar} FR:{fr_bar} | {phase_text}")

        if terminated:
            print("\n-------------------------------------------------------")
            print(f"OUTCOME: {info['event'].upper()} at t={step * env.dt:.2f}s!")
            print(f"Final Altitude: {env.state[2]:.2f} m, Final Vz: {env.state[5]:.2f} m/s, Final Roll: {math.degrees(env.state[6]):.1f} deg")
            print(f"Total Cumulative Reward: {total_reward:.1f}")
            print("-------------------------------------------------------\n")
            break

        time.sleep(0.03)


def run_stunt2_demo():
    """
    Demonstrates Stunt 2: Obstacle Leap.
    Vehicle approaches obstacle, bursts thrusters for altitude, clears truck, and cushions landing.
    """
    print("\n=======================================================")
    print("DEMO: STUNT 2 - OBSTACLE TRUCK LEAP")
    print("Obstacle located at x = 0m (Height = 3.8m, Length = 8.0m).")
    print("Vehicle begins at x = -18m traveling forward at 12 m/s.")
    print("Watch thruster vectoring hop over the truck and land!")
    print("=======================================================\n")

    env = RoadsterStuntsEnv(mode="leap", dt=0.04)
    obs, info = env.reset(seed=202)

    total_reward = 0.0
    for step in range(120):
        x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = env.state

        action = np.zeros(4, dtype=np.float32)

        # 3 Stages of leap maneuver:
        if x < -5.0:
            action = np.full(4, 0.95, dtype=np.float32)
            phase_text = "BOOST TAKEOFF"
        elif x <= 5.0:
            action = np.full(4, 0.45, dtype=np.float32)
            phase_text = "FLYING OVER OBSTACLE"
        else:
            brake = max(0.0, (-vz - 1.0) * 0.25)
            action = np.clip(np.full(4, 0.40 + brake, dtype=np.float32), 0.0, 1.0)
            phase_text = "TOUCHDOWN CUSHION"

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 3 == 0 or terminated:
            t_sec = step * env.dt
            alt_bar = make_bar(env.state[2], max_val=8.0, length=8)
            x_pos = env.state[0]

            in_truck_zone = "[OVER TRUCK]" if abs(x_pos) <= 4.0 else "            "

            print(f"t={t_sec:4.2f}s | X:{x_pos:5.1f}m | Alt:{env.state[2]:4.1f}m {alt_bar} | Vz:{env.state[5]:5.1f}m/s | "
                  f"{in_truck_zone} | {phase_text}")

        if terminated:
            print("\n-------------------------------------------------------")
            print(f"OUTCOME: {info['event'].upper()} at t={step * env.dt:.2f}s!")
            print(f"Final X Position: {env.state[0]:.2f} m (Cleared truck at x=0m)")
            print(f"Final Altitude: {env.state[2]:.2f} m, Final Vz: {env.state[5]:.2f} m/s")
            print(f"Total Cumulative Reward: {total_reward:.1f}")
            print("-------------------------------------------------------\n")
            break

        time.sleep(0.03)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "1":
        run_stunt1_demo()
    elif arg == "2":
        run_stunt2_demo()
    else:
        print("Choose a stunt demo to run:")
        print("  1: Stunt 1 - Upside-Down Inverted Drop (Mid-Air 180-deg Flip)")
        print("  2: Stunt 2 - Obstacle Leap (Flyover & Touchdown)")
        print("  all: Run both stunts sequentially")
        
        print("\n[Auto-running both stunt demonstrations]...\n")
        run_stunt1_demo()
        time.sleep(1.0)
        run_stunt2_demo()
