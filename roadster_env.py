import math
import numpy as np

# We try importing gymnasium first, but if it is not installed yet,
# we provide a clean lightweight fallback so our code runs anywhere without breaking.
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Minimal fallback class for Gym Environment
    class gym:
        class Env:
            pass

    # Minimal fallback for spaces.Box
    class spaces:
        class Box:
            def __init__(self, low, high, shape=None, dtype=np.float32):
                self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.array(low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.array(high, dtype=dtype)
                self.shape = shape
                self.dtype = dtype

            def sample(self):
                # Generates a random sample inside the bounded box
                return np.random.uniform(self.low, self.high).astype(self.dtype)


class RoadsterStuntsEnv(gym.Env):
    """
    Simulation environment for the Tesla Roadster 4-thruster landing stunts.
    It supports two main stunts:
    1) 'flip': Dropped upside down, car must flip 180 degrees upright and land on wheels.
    2) 'leap': Moving forward, car must fire thrusters to fly over a Tesla Semi truck and land safely.
    
    Physics includes pure gravity and 6-DoF rigid body dynamics,
    with no aerodynamic drag or buoyant forces, keeping it clean and easy to understand.
    """

    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, mode="flip", dt=0.02):
        super().__init__()

        # Mode can be 'flip' or 'leap'
        self.mode = mode
        self.dt = dt  # Physics step time, 50 Hz simulation (0.02 seconds)

        # Vehicle physical parameters (modeled as a sporty electric roadster)
        self.mass = 1500.0  # Mass in kg
        self.length = 4.5   # Length in meters
        self.width = 2.0    # Width in meters
        self.height = 1.2   # Height in meters
        self.gravity = 9.81 # Acceleration due to gravity in m/s^2

        # Moments of inertia (using standard rectangular box approximation)
        # Ixx is for roll, Iyy is for pitch, Izz is for yaw
        self.Ixx = (1.0 / 12.0) * self.mass * (self.width**2 + self.height**2)
        self.Iyy = (1.0 / 12.0) * self.mass * (self.length**2 + self.height**2)
        self.Izz = (1.0 / 12.0) * self.mass * (self.length**2 + self.width**2)

        # Thruster setup: 4 downward-firing thrusters placed at the 4 corners of the car
        # Maximum thrust per thruster in Newtons
        # Total max thrust gives a thrust-to-weight ratio of about 2.5
        self.max_thrust_per_nozzle = 9200.0  # Newtons

        # Half dimensions for torque lever arms
        self.half_L = self.length / 2.0
        self.half_W = self.width / 2.0

        # Obstacle parameters (Tesla Semi truck bounding box for the leap stunt)
        # Position is centered at x=0, y=0, z=0 on the runway
        self.truck_x = 0.0
        self.truck_y = 0.0
        self.truck_length = 8.0  # Length along X axis in meters
        self.truck_width = 2.6   # Width along Y axis in meters
        self.truck_height = 3.8  # Height along Z axis in meters

        # Action space: 4 continuous throttle values in [0, 1]
        # [Front-Left, Front-Right, Rear-Left, Rear-Right]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Observation space: 15 continuous values
        # [x, y, z, vx, vy, vz, sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw, wx, wy, wz]
        # Using sin and cos of angles avoids angle wrap around issues at pi and -pi
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )

        # Episode tracking variables
        self.state = None
        self.current_step = 0
        self.max_steps = 600 # 12 seconds of simulated time per episode

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial condition based on the chosen mode.
        """
        super().reset(seed=seed) if hasattr(super(), "reset") else None
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        if self.mode == "flip":
            # Stunt 1: Upside-down drop
            # Start high up at z = 25m, with roll angle near 180 degrees (upside down)
            # Add a small random tilt and small angular velocity to make it interesting
            x0 = np.random.uniform(-1.0, 1.0)
            y0 = np.random.uniform(-1.0, 1.0)
            z0 = 25.0 + np.random.uniform(-1.0, 1.0)

            vx0 = np.random.uniform(-0.5, 0.5)
            vy0 = np.random.uniform(-0.5, 0.5)
            vz0 = np.random.uniform(-1.0, 0.0) # Starting with a slight downward speed

            # Upside down means roll angle is approximately pi (180 degrees)
            roll0 = math.pi + np.random.uniform(-0.3, 0.3)
            pitch0 = np.random.uniform(-0.2, 0.2)
            yaw0 = np.random.uniform(-0.2, 0.2)

            # Small initial tumbling angular rate
            wx0 = np.random.uniform(-0.5, 0.5)
            wy0 = np.random.uniform(-0.5, 0.5)
            wz0 = 0.0

        elif self.mode == "leap":
            # Stunt 2: Semi Truck Leap
            # Start behind the truck at x = -18m, at normal driving level z = 1.0m
            # Moving forward toward the truck with initial forward speed
            x0 = -18.0 + np.random.uniform(-0.5, 0.5)
            y0 = np.random.uniform(-0.2, 0.2)
            z0 = 1.0

            vx0 = 12.0 + np.random.uniform(-0.5, 0.5) # Moving forward at around 12 m/s
            vy0 = 0.0
            vz0 = 0.0

            # Upright attitude
            roll0 = 0.0
            pitch0 = 0.0
            yaw0 = 0.0

            wx0 = 0.0
            wy0 = 0.0
            wz0 = 0.0
        else:
            raise ValueError(f"Unknown mode: {self.mode}, must be 'flip' or 'leap'")

        # Internal state vector has 12 components:
        # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        self.state = np.array([
            x0, y0, z0,
            vx0, vy0, vz0,
            roll0, pitch0, yaw0,
            wx0, wy0, wz0
        ], dtype=np.float32)

        obs = self._get_obs()
        info = {"mode": self.mode}
        return obs, info

    def _get_obs(self):
        """
        Converts internal 12D state into a 15D observation vector with trigonometric angles.
        This helps neural networks learn smooth continuous representations without angle jump.
        """
        x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = self.state

        obs = np.array([
            x, y, z,
            vx, vy, vz,
            math.sin(roll), math.cos(roll),
            math.sin(pitch), math.cos(pitch),
            math.sin(yaw), math.cos(yaw),
            wx, wy, wz
        ], dtype=np.float32)
        return obs

    def step(self, action):
        """
        Advances the simulation by dt using the 4 thruster inputs.
        """
        self.current_step += 1

        # Clip action between 0 and 1 to be completely safe
        action = np.clip(action, 0.0, 1.0)
        t_fl, t_fr, t_rl, t_rr = action

        # Compute individual thruster forces in Newtons
        f_fl = t_fl * self.max_thrust_per_nozzle
        f_fr = t_fr * self.max_thrust_per_nozzle
        f_rl = t_rl * self.max_thrust_per_nozzle
        f_rr = t_rr * self.max_thrust_per_nozzle

        # Unpack current state
        x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = self.state

        # Compute torques around body axes using moment arms
        # Roll torque: left side thrust pushes left up, right side thrust pushes right up
        tau_x = self.half_W * ((f_fl + f_rl) - (f_fr + f_rr))

        # Pitch torque: rear thrust pushes back up (nose down), front thrust pushes front up (nose up)
        # Using front minus rear gives positive pitch torque when front fires more
        tau_y = self.half_L * ((f_fl + f_fr) - (f_rl + f_rr))

        # Yaw torque is zero because all 4 thrusters fire purely along the car's local vertical axis
        tau_z = 0.0

        # Angular accelerations from Euler's rotational equations
        alpha_x = tau_x / self.Ixx
        alpha_y = tau_y / self.Iyy
        alpha_z = tau_z / self.Izz

        # Update angular velocities
        wx_new = wx + alpha_x * self.dt
        wy_new = wy + alpha_y * self.dt
        wz_new = wz + alpha_z * self.dt

        # Update Euler angles
        roll_new = roll + wx_new * self.dt
        pitch_new = pitch + wy_new * self.dt
        yaw_new = yaw + wz_new * self.dt

        # Keep angles normalized inside [-pi, pi] for clean math
        roll_new = (roll_new + math.pi) % (2.0 * math.pi) - math.pi
        pitch_new = (pitch_new + math.pi) % (2.0 * math.pi) - math.pi
        yaw_new = (yaw_new + math.pi) % (2.0 * math.pi) - math.pi

        # Compute thrust direction in world coordinates
        # Thrusters push upward along the car's local roof axis (local +Z)
        # Using standard rotation matrix column for local Z:
        # u_x = sin(roll)*sin(yaw) + cos(roll)*sin(pitch)*cos(yaw)
        # u_y = -sin(roll)*cos(yaw) + cos(roll)*sin(pitch)*sin(yaw)
        # u_z = cos(roll)*cos(pitch)
        cr = math.cos(roll_new)
        sr = math.sin(roll_new)
        cp = math.cos(pitch_new)
        sp = math.sin(pitch_new)
        cy = math.cos(yaw_new)
        sy = math.sin(yaw_new)

        u_x = sr * sy + cr * sp * cy
        u_y = -sr * cy + cr * sp * sy
        u_z = cr * cp

        # Total magnitude of thrust force from all 4 nozzles
        total_thrust = f_fl + f_fr + f_rl + f_rr

        # Acceleration in world frame (thrust force divided by mass, minus gravity in z)
        ax = (total_thrust * u_x) / self.mass
        ay = (total_thrust * u_y) / self.mass
        az = (total_thrust * u_z) / self.mass - self.gravity

        # Update linear velocities using semi-implicit Euler integration
        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        vz_new = vz + az * self.dt

        # Update positions
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        z_new = z + vz_new * self.dt

        # Save new state
        self.state = np.array([
            x_new, y_new, z_new,
            vx_new, vy_new, vz_new,
            roll_new, pitch_new, yaw_new,
            wx_new, wy_new, wz_new
        ], dtype=np.float32)

        # Check contact, collisions, and termination conditions
        terminated, truncated, event = self._check_termination(action)
        reward = self._compute_reward(action, event)

        obs = self._get_obs()
        info = {
            "event": event,
            "thrust_fl": f_fl,
            "thrust_fr": f_fr,
            "thrust_rl": f_rl,
            "thrust_rr": f_rr
        }

        return obs, reward, terminated, truncated, info

    def _check_termination(self, action):
        """
        Evaluates whether the episode has finished due to landing, crash, or obstacle collision.
        """
        x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = self.state

        # Check maximum time limit truncation
        if self.current_step >= self.max_steps:
            return False, True, "timeout"

        # Check if the car hit the obstacle (Tesla Semi truck) during the leap stunt
        if self.mode == "leap":
            half_t_len = self.truck_length / 2.0
            half_t_wid = self.truck_width / 2.0
            # Collision happens if car is inside the truck's 3D bounding box
            in_x = (self.truck_x - half_t_len) <= x <= (self.truck_x + half_t_len)
            in_y = (self.truck_y - half_t_wid) <= y <= (self.truck_y + half_t_wid)
            in_z = z <= self.truck_height
            if in_x and in_y and in_z:
                return True, False, "truck_collision"

        # Check ground contact condition (wheels touch runway at z <= 0.6m, half car height)
        ground_level = self.height / 2.0
        if z <= ground_level:
            # Landing criteria: upright roll and pitch, low vertical speed, low angular rates
            is_upright = (abs(roll) < math.radians(15.0)) and (abs(pitch) < math.radians(15.0))
            is_gentle_speed = abs(vz) < 2.5
            is_stable_spin = (abs(wx) < 1.0) and (abs(wy) < 1.0)

            if is_upright and is_gentle_speed and is_stable_spin:
                # If in leap mode, also verify it passed the truck (x > truck_x + half_t_len)
                if self.mode == "leap":
                    if x > (self.truck_x + self.truck_length / 2.0):
                        return True, False, "successful_landing"
                    else:
                        return True, False, "landed_too_early"
                return True, False, "successful_landing"
            else:
                return True, False, "crash_landing"

        # Out of bounds check: if car flies excessively high or drifts off the airfield
        if z > 80.0 or abs(x) > 200.0 or abs(y) > 200.0:
            return True, False, "out_of_bounds"

        return False, False, "in_flight"

    def _compute_reward(self, action, event):
        """
        Reward function designed for stability, gentle descent, and clean landing.
        """
        x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = self.state

        # Terminal events give clear, decisive feedback
        if event == "successful_landing":
            return 250.0
        elif event == "crash_landing":
            return -150.0
        elif event == "truck_collision":
            return -200.0
        elif event == "out_of_bounds":
            return -100.0
        elif event == "landed_too_early":
            return -80.0

        # Step rewards while in flight
        # 1) Reward for staying upright (penalize roll and pitch deviations from zero)
        r_upright = -1.5 * (roll**2 + pitch**2)

        # 2) Reward for damping angular spin rates
        r_spin = -0.2 * (wx**2 + wy**2 + wz**2)

        # 3) Fuel penalty to encourage efficient thruster firing
        r_fuel = -0.05 * np.sum(action**2)

        if self.mode == "flip":
            # For the flip stunt, reward decreasing altitude gently towards ground
            r_height = -0.1 * z
            # Penalize excessive downward falling speed
            r_speed = -0.1 * max(0.0, -vz - 5.0)**2
            return r_upright + r_spin + r_height + r_speed + r_fuel

        elif self.mode == "leap":
            # For the leap stunt, reward moving forward towards the target past the truck
            r_progress = 0.5 * vx
            # Clearance bonus when safely above the truck height near x=0
            r_clearance = 0.0
            if abs(x - self.truck_x) < (self.truck_length / 2.0 + 2.0):
                if z > (self.truck_height + 1.0):
                    r_clearance = 2.0  # Safe altitude above truck
                else:
                    r_clearance = -3.0 # Dangerously close to truck top

            return r_upright + r_spin + r_progress + r_clearance + r_fuel

        return 0.0

