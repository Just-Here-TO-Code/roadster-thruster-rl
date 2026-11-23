import math
import numpy as np
from roadster_env import RoadsterStuntsEnv

def test_free_fall():
    """
    Test 1: Free fall under pure gravity with zero thruster firing.
    Verifies that acceleration matches g = 9.81 m/s^2 downwards.
    """
    print("[Test 1] Testing Free Fall Dynamics...")
    env = RoadsterStuntsEnv(mode="flip", dt=0.01)
    env.reset(seed=42)

    # Set known initial state: z = 30m, stationary, perfectly upright
    env.state = np.array([0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    zero_action = np.zeros(4, dtype=np.float32)
    steps = 50 # 0.50 seconds
    for _ in range(steps):
        env.step(zero_action)

    expected_vz = -env.gravity * (steps * env.dt)
    expected_z = 30.0 - 0.5 * env.gravity * ((steps * env.dt) ** 2)

    sim_z = env.state[2]
    sim_vz = env.state[5]

    print(f"   Theoretical vz: {expected_vz:.4f} m/s, Simulated vz: {sim_vz:.4f} m/s")
    print(f"   Theoretical z:  {expected_z:.4f} m,   Simulated z:  {sim_z:.4f} m")

    # Tolerances allow for first-order Euler integration difference
    assert abs(sim_vz - expected_vz) < 0.1, "Free fall velocity mismatch"
    assert abs(sim_z - expected_z) < 0.2, "Free fall altitude mismatch"
    print("   -> PASS: Free fall under gravity behaves as expected.\n")


def test_symmetrical_hover():
    """
    Test 2: Perfect hover balance.
    When 4 thrusters exert total force equal to vehicle weight,
    vertical acceleration should be zero and the car should hover.
    """
    print("[Test 2] Testing Symmetrical Hover Equilibrium...")
    env = RoadsterStuntsEnv(mode="flip", dt=0.01)
    env.reset(seed=42)

    # Total weight W = m * g
    weight = env.mass * env.gravity
    hover_throttle_each = weight / (4.0 * env.max_thrust_per_nozzle)

    # Place car at z = 15m, stationary, upright
    env.state = np.array([0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    hover_action = np.full(4, hover_throttle_each, dtype=np.float32)
    for _ in range(100): # 1.0 second of simulation
        env.step(hover_action)

    sim_z = env.state[2]
    sim_vz = env.state[5]

    print(f"   Hover throttle per nozzle: {hover_throttle_each * 100:.2f}%")
    print(f"   Altitude after 1s hover: {sim_z:.4f} m (Initial: 15.0000 m)")
    print(f"   Vertical velocity after 1s: {sim_vz:.4f} m/s")

    assert abs(sim_z - 15.0) < 0.05, "Hover altitude drifted significantly"
    assert abs(sim_vz) < 0.05, "Hover vertical velocity is not near zero"
    print("   -> PASS: Vehicle maintains stable hover at constant altitude.\n")


def test_differential_torque():
    """
    Test 3: Differential torque generation.
    Firing the left thrusters more than the right thrusters must produce
    a positive roll torque and clockwise angular acceleration.
    """
    print("[Test 3] Testing Differential Torque and Attitude Control...")
    env = RoadsterStuntsEnv(mode="flip", dt=0.01)
    env.reset(seed=42)

    # Start upright and motionless
    env.state = np.array([0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Throttle: Left thrusters fire at 80%, right thrusters fire at 0%
    # Action format: [FL, FR, RL, RR]
    roll_action = np.array([0.8, 0.0, 0.8, 0.0], dtype=np.float32)

    for _ in range(10):
        env.step(roll_action)

    roll_rate_wx = env.state[9]
    roll_angle = env.state[6]

    print(f"   Roll rate (wx) after asymmetric thrust: {roll_rate_wx:.4f} rad/s")
    print(f"   Roll angle (phi) after asymmetric thrust: {math.degrees(roll_angle):.2f} degrees")

    assert roll_rate_wx > 0.5, "Asymmetric thrust failed to produce roll angular velocity"
    print("   -> PASS: Corner thrusters generate correct attitude torque.\n")


def test_truck_collision_logic():
    """
    Test 4: Tesla Semi truck collision detection during leap mode.
    Verifies that the bounding box logic flags a collision when intersecting the truck.
    """
    print("[Test 4] Testing Truck Obstacle Collision Detection...")
    env = RoadsterStuntsEnv(mode="leap", dt=0.02)
    env.reset(seed=42)

    # Place car directly inside the truck bounding box at x=0, y=0, z=2.0m (truck height is 3.8m)
    env.state = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    dummy_action = np.zeros(4, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(dummy_action)

    print(f"   Termination event: {info['event']}, Terminated flag: {terminated}, Reward: {reward}")

    assert terminated is True, "Collision was not flagged as terminated"
    assert info["event"] == "truck_collision", "Event name mismatch"
    assert reward <= -150.0, "Collision penalty was not applied"
    print("   -> PASS: Obstacle bounding box collision correctly detected.\n")


def test_gym_interface_and_shapes():
    """
    Test 5: Gymnasium interface compatibility.
    Checks observation and action space shapes, step return values, and mode switching.
    """
    print("[Test 5] Testing Gymnasium Interface and Observation Shapes...")
    for mode in ["flip", "leap"]:
        env = RoadsterStuntsEnv(mode=mode)
        obs, info = env.reset()

        assert obs.shape == (15,), f"Obs shape is {obs.shape}, expected (15,)"
        assert len(info) > 0, "Info dictionary is empty"

        # Take 5 random steps
        for _ in range(5):
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(random_action)
            assert obs.shape == (15,), "Obs shape changed during step"
            assert isinstance(reward, (float, np.floating)), "Reward must be a float"

        print(f"   -> Mode '{mode}' passed reset and random step tests.")
    print("   -> PASS: Gymnasium API is fully compliant.\n")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING VERIFICATION TESTS FOR ROADSTER SIMULATION")
    print("==================================================\n")

    test_free_fall()
    test_symmetrical_hover()
    test_differential_torque()
    test_truck_collision_logic()
    test_gym_interface_and_shapes()

    print("==================================================")
    print("ALL 5 DYNAMICS TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

