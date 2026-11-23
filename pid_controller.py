import math
import numpy as np

class PIDController:
    """
    A clean, intuitive single-axis PID controller.
    It calculates corrective output based on proportional error,
    accumulated integral error, and rate of change (derivative).
    """

    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0)):
        self.kp = kp  # Proportional gain, reacts to present error
        self.ki = ki  # Integral gain, reacts to past accumulated error
        self.kd = kd  # Derivative gain, dampens future overshoot

        self.min_out, self.max_out = output_limits

        self.integral = 0.0
        self.prev_error = 0.0
        self.has_prev = False

    def reset(self):
        """
        Clears accumulated integral and past derivative states.
        """
        self.integral = 0.0
        self.prev_error = 0.0
        self.has_prev = False

    def update(self, error, dt, derivative=None):
        """
        Computes the PID output for a given error and time step.
        If an explicit derivative is provided (like angular velocity),
        we use it directly instead of finite difference for better stability.
        """
        # Proportional term
        p_term = self.kp * error

        # Integral term with tight anti-windup clamping to prevent lag
        self.integral += error * dt
        self.integral = max(-0.2, min(0.2, self.integral))
        i_term = self.ki * self.integral

        # Derivative term
        if derivative is not None:
            # If physical rate is already known, use it directly
            d_term = -self.kd * derivative
        elif self.has_prev and dt > 0:
            d_error = (error - self.prev_error) / dt
            d_term = self.kd * d_error
        else:
            d_term = 0.0

        self.prev_error = error
        self.has_prev = True

        total_output = p_term + i_term + d_term
        # Clamp output to specified limits
        return max(self.min_out, min(self.max_out, total_output))


class RoadsterPIDAgent:
    """
    Classical baseline controller for the Tesla Roadster lander.
    It uses 3 separate loops:
    1) Roll PID: balances the car left to right
    2) Pitch PID: balances the car nose to tail
    3) Altitude/Descent PD: manages vertical deceleration with gravity feedforward
    """

    def __init__(self, env):
        self.env = env

        # Hover throttle equilibrium where total upward thrust equals vehicle weight
        # Weight = 1500kg * 9.81 = 14,715 N
        # Total max thrust = 4 * 9200 = 36,800 N
        # Hover throttle = 14,715 / 36,800 ~ 0.3998 (about 40%)
        self.hover_throttle = (env.mass * env.gravity) / (4.0 * env.max_thrust_per_nozzle)

        # PID controllers for attitude stabilization
        # Roll PID: error is -roll_angle, derivative is roll_rate wx
        self.roll_pid = PIDController(kp=2.0, ki=0.02, kd=0.7, output_limits=(-0.4, 0.4))

        # Pitch PID: error is -pitch_angle, derivative is pitch_rate wy
        self.pitch_pid = PIDController(kp=2.4, ki=0.02, kd=0.8, output_limits=(-0.4, 0.4))

        # Vertical velocity descent PD: pure PD with gravity feedforward avoids integral windup
        self.descent_pid = PIDController(kp=0.30, ki=0.0, kd=0.02, output_limits=(-0.4, 0.55))

    def reset(self):
        """
        Resets all internal PID controller states before a new run.
        """
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.descent_pid.reset()

    def get_action(self, state, mode="flip"):
        """
        Takes the 12D physical state and returns 4 thruster throttles in [0, 1].
        State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        """
        x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = state
        dt = self.env.dt

        # 1) Roll control: target roll is 0 (level)
        # Error is target - current = -roll
        roll_error = -roll
        # Inverted angle wrapping adjustment so it always turns the shorter way
        if roll_error > math.pi:
            roll_error -= 2.0 * math.pi
        elif roll_error < -math.pi:
            roll_error += 2.0 * math.pi

        delta_roll = self.roll_pid.update(error=roll_error, dt=dt, derivative=wx)

        # 2) Pitch control: target pitch is 0 (level)
        pitch_error = -pitch
        delta_pitch = self.pitch_pid.update(error=pitch_error, dt=dt, derivative=wy)

        # 3) Vertical velocity control
        # Target descent rate: smooth glide down to a safe touchdown speed of 1.6 m/s
        target_vz = -max(1.6, min(5.5, 0.40 * z))
        vz_error = target_vz - vz
        delta_z = self.descent_pid.update(error=vz_error, dt=dt)

        # Base throttle is set to hover throttle plus vertical adjustment
        base_throttle = self.hover_throttle + delta_z

        # 4) Thruster Mixer (combining vertical lift with roll and pitch torques)
        # Front-Left:  base + delta_roll + delta_pitch
        # Front-Right: base - delta_roll + delta_pitch
        # Rear-Left:   base + delta_roll - delta_pitch
        # Rear-Right:  base - delta_roll - delta_pitch
        #
        # Sign logic:
        # If car tilts right (roll > 0), delta_roll is negative,
        # so right thrusters get higher throttle, pushing right side up.
        # If nose tilts up (pitch > 0), delta_pitch is negative,
        # so rear thrusters get higher throttle, pushing rear side up.
        t_fl = base_throttle + delta_roll + delta_pitch
        t_fr = base_throttle - delta_roll + delta_pitch
        t_rl = base_throttle + delta_roll - delta_pitch
        t_rr = base_throttle - delta_roll - delta_pitch

        # Clamp all 4 throttles safely between 0.0 and 1.0
        action = np.clip(np.array([t_fl, t_fr, t_rl, t_rr], dtype=np.float32), 0.0, 1.0)
        return action
