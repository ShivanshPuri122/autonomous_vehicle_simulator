import numpy as np

class PIDController:
    """
    Standard PID Controller implementation.
    """
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0):
        self._k_p = k_p
        self._k_i = k_i
        self._k_d = k_d
        self._error_sum = 0.0
        self._last_error = 0.0

    def compute(self, error, dt):
        """
        Computes the control output based on the error and time step.
        """
        if dt <= 0:
            return 0.0
            
        self._error_sum += error * dt
        error_diff = (error - self._last_error) / dt
        self._last_error = error
        
        return (self._k_p * error) + (self._k_i * self._error_sum) + (self._k_d * error_diff)

    def reset(self):
        """Resets the integral and derivative terms."""
        self._error_sum = 0.0
        self._last_error = 0.0


class LongitudinalController:
    """
    Handles speed control (Throttle/Brake).
    """
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0):
        self.pid = PIDController(k_p, k_i, k_d)

    def run_step(self, target_speed, current_speed, dt):
        """
        Calculates the throttle value [0.0, 1.0] to reach target speed.
        """
        error = target_speed - current_speed
        throttle = self.pid.compute(error, dt)
        return np.clip(throttle, 0.0, 1.0)


class LateralController:
    """
    Handles steering control to maintain lane centering.
    """
    def __init__(self, k_p=0.5, k_i=0.01, k_d=0.1):
        self.pid = PIDController(k_p, k_i, k_d)

    def run_step(self, lane_offset, dt):
        """
        Calculates the steering angle [-1.0, 1.0] based on lane offset.
        An offset of 0.0 means the vehicle is perfectly centered.
        """
        # We negate the offset because if the vehicle is to the right (+offset), 
        # we need to steer left (-steering).
        steering = self.pid.compute(-lane_offset, dt)
        return np.clip(steering, -1.0, 1.0)
