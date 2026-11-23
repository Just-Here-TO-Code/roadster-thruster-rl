import math
import json
import numpy as np
from roadster_env import RoadsterStuntsEnv
from pid_controller import RoadsterPIDAgent
from benchmark_rl_vs_pid import get_rl_action

def record_trajectory(env, agent_type="rl", mode="flip", initial_state=None):
    """
    Simulates a full episode and records high-frequency telemetry at each time step.
    agent_type can be 'rl' or 'pid'.
    mode can be 'flip' or 'leap'.
    """
    env.mode = mode
    env.max_steps = 600
    env.reset(seed=42)

    if initial_state is not None:
        env.state = np.array(initial_state, dtype=np.float32)

    pid_agent = RoadsterPIDAgent(env) if agent_type == "pid" else None

    frames = []
    total_reward = 0.0

    for step in range(env.max_steps):
        # Current state snapshot
        x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = env.state

        # Get action from the selected agent
        if agent_type == "pid":
            action = pid_agent.get_action(env.state, mode=mode)
        else:
            action = get_rl_action(env.state, mode=mode)

        # Store telemetry frame before stepping physics
        frame = {
            "step": step,
            "t": round(step * env.dt, 3),
            "pos": [round(float(x), 3), round(float(y), 3), round(float(z), 3)],
            "vel": [round(float(vx), 3), round(float(vy), 3), round(float(vz), 3)],
            "rot": [
                round(float(math.degrees(roll)), 2),
                round(float(math.degrees(pitch)), 2),
                round(float(math.degrees(yaw)), 2)
            ],
            "rot_rad": [round(float(roll), 4), round(float(pitch), 4), round(float(yaw), 4)],
            "omega": [round(float(wx), 3), round(float(wy), 3), round(float(wz), 3)],
            "throttles": [
                round(float(action[0]), 3),
                round(float(action[1]), 3),
                round(float(action[2]), 3),
                round(float(action[3]), 3)
            ]
        }

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame["event"] = info.get("event", "in_flight")
        frames.append(frame)

        if terminated or truncated:
            break

    summary = {
        "agent": agent_type.upper(),
        "mode": mode,
        "total_frames": len(frames),
        "duration_sec": round(len(frames) * env.dt, 2),
        "final_event": info.get("event", "timeout"),
        "final_pos": [round(float(env.state[0]), 2), round(float(env.state[1]), 2), round(float(env.state[2]), 2)],
        "final_vz": round(float(env.state[5]), 2),
        "final_tilt_deg": round(float(math.degrees(math.sqrt(env.state[6]**2 + env.state[7]**2))), 1),
        "total_reward": round(float(total_reward), 1)
    }

    return {"summary": summary, "frames": frames}


def export_all_trajectories():
    """
    Generates 4 flight trajectories demonstrating RL vs PID across both stunts,
    and writes them to trajectories.js for instant browser playback.
    """
    print("=========================================================================================")
    print("EXPORTING HIGH-FIDELITY FLIGHT TRAJECTORIES FOR THREE.JS VISUALIZER")
    print("=========================================================================================\n")

    env = RoadsterStuntsEnv()

    # 1) Stunt 1: 180-degree Inverted Drop with RL Agent
    print("[1/4] Simulating Stunt 1 (180 deg Inverted Drop) with RL Agent...")
    init_flip = [0.0, 0.0, 25.0, 0.0, 0.0, -1.0, math.pi, 0.0, 0.0, 0.0, 0.0, 0.0]
    traj_s1_rl = record_trajectory(env, agent_type="rl", mode="flip", initial_state=init_flip)
    print(f"      Outcome: {traj_s1_rl['summary']['final_event']} in {traj_s1_rl['summary']['duration_sec']}s (Landing Vz: {traj_s1_rl['summary']['final_vz']} m/s)")

    # 2) Stunt 1: 180-degree Inverted Drop with PID Controller
    print("\n[2/4] Simulating Stunt 1 (180 deg Inverted Drop) with Classical PID...")
    traj_s1_pid = record_trajectory(env, agent_type="pid", mode="flip", initial_state=init_flip)
    print(f"      Outcome: {traj_s1_pid['summary']['final_event']} in {traj_s1_pid['summary']['duration_sec']}s (Landing Vz: {traj_s1_pid['summary']['final_vz']} m/s)")

    # 3) Stunt 2: Tesla Semi Leap with RL Agent
    print("\n[3/4] Simulating Stunt 2 (Tesla Semi Leap) with RL Agent...")
    init_leap = [-18.0, 0.0, 1.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    traj_s2_rl = record_trajectory(env, agent_type="rl", mode="leap", initial_state=init_leap)
    print(f"      Outcome: {traj_s2_rl['summary']['final_event']} in {traj_s2_rl['summary']['duration_sec']}s (Final X: {traj_s2_rl['summary']['final_pos'][0]} m)")

    # 4) Stunt 2: Tesla Semi Leap with Classical PID
    print("\n[4/4] Simulating Stunt 2 (Tesla Semi Leap) with Classical PID...")
    traj_s2_pid = record_trajectory(env, agent_type="pid", mode="leap", initial_state=init_leap)
    print(f"      Outcome: {traj_s2_pid['summary']['final_event']} in {traj_s2_pid['summary']['duration_sec']}s (Final X: {traj_s2_pid['summary']['final_pos'][0]} m)")

    dataset = {
        "stunt1_rl": traj_s1_rl,
        "stunt1_pid": traj_s1_pid,
        "stunt2_rl": traj_s2_rl,
        "stunt2_pid": traj_s2_pid,
        "environment": {
            "car_length": env.length,
            "car_width": env.width,
            "car_height": env.height,
            "truck_x": env.truck_x,
            "truck_y": env.truck_y,
            "truck_length": env.truck_length,
            "truck_width": env.truck_width,
            "truck_height": env.truck_height
        }
    }

    # Save as JSON
    json_path = "trajectories.json"
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nSaved raw JSON to '{json_path}'.")

    # Also save as a JavaScript file so index.html can load it locally via script tag without CORS errors!
    js_path = "trajectories.js"
    with open(js_path, "w") as f:
        f.write("// Flight telemetry dataset generated by export_trajectories.py\n")
        f.write("window.ROADSTER_TELEMETRY = ")
        json.dump(dataset, f)
        f.write(";\n")
    print(f"Saved browser-ready dataset to '{js_path}'.\n")


if __name__ == "__main__":
    export_all_trajectories()
