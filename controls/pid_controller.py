# pyrefly: ignore [missing-import]
"""
=============================================================================
PID Controllers for Autonomous Vehicle Control
=============================================================================

This module provides separate Longitudinal and Lateral PID controllers
optimized for real-time performance in the CARLA simulator.

Key Features:
    - Anti-windup clamping on the integral term
    - Exponential Moving Average (EMA) smoothing on the derivative term
    - Steering rate limiter for smooth turning transitions
    - Configurable output clamping
    - Built-in logging and debugging utilities
    - Safety checks for NaN / infinite inputs

Author : Controls Team
Version: 1.0.0
"""

import math
import logging
import time

# ── Module-level logger ─────────────────────────────────────────────────────
logger = logging.getLogger("controls.pid")


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER: safe division to avoid ZeroDivisionError
# ═════════════════════════════════════════════════════════════════════════════
def _safe_div(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, or 0.0 when denominator is near zero."""
    if abs(denominator) < 1e-9:
        return 0.0
    return numerator / denominator


# ═════════════════════════════════════════════════════════════════════════════
#  LONGITUDINAL PID — throttle / brake control
# ═════════════════════════════════════════════════════════════════════════════
class LongitudinalPID:
    """
    PID controller for longitudinal (speed) control.

    Converts a speed error (target_speed − current_speed) into throttle
    and brake commands that can be directly applied to a CARLA vehicle.

    Tuning Parameters (constructor args)
    -------------------------------------
    K_P : float   – Proportional gain.  Start ≈ 1.0.
    K_I : float   – Integral gain.      Start ≈ 0.05–0.1.
    K_D : float   – Derivative gain.    Start ≈ 0.01–0.05.
    dt  : float   – Expected time-step in seconds (match CARLA fixed step).

    Advanced Settings (attributes)
    ------------------------------
    integral_max : float – Anti-windup clamp for the integral accumulator.
    ema_alpha    : float – EMA smoothing factor for derivative (0 = full
                           smoothing, 1 = no smoothing).
    """

    def __init__(
        self,
        K_P: float = 1.0,
        K_I: float = 0.05,
        K_D: float = 0.01,
        dt: float = 0.05,
    ):
        # ── Gains ────────────────────────────────────────────────────────
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.dt = dt

        # ── Internal state ───────────────────────────────────────────────
        self._error_integral: float = 0.0
        self._error_derivative: float = 0.0
        self._previous_error: float = 0.0

        # ── Anti-windup clamp ────────────────────────────────────────────
        #    Prevents the integral term from growing unbounded when the
        #    vehicle is stuck (e.g., at a red light for a long time).
        self.integral_max: float = 5.0

        # ── Derivative EMA smoothing ─────────────────────────────────────
        #    A value of 1.0 means no smoothing (raw derivative).
        #    Lower values provide heavier smoothing of sensor noise.
        self.ema_alpha: float = 0.4

        # ── Debug bookkeeping ────────────────────────────────────────────
        self._last_P: float = 0.0
        self._last_I: float = 0.0
        self._last_D: float = 0.0

    # ------------------------------------------------------------------
    #  Main computation
    # ------------------------------------------------------------------
    def run_step(self, target_speed: float, current_speed: float):
        """
        Compute throttle and brake from a speed set-point.

        Parameters
        ----------
        target_speed  : float – Desired speed in m/s (≥ 0).
        current_speed : float – Measured speed in m/s.

        Returns
        -------
        (throttle, brake) : tuple[float, float]
            throttle ∈ [0.0, 1.0], brake ∈ [0.0, 1.0].
        """
        # ── Safety: reject garbage inputs ────────────────────────────────
        if math.isnan(target_speed) or math.isnan(current_speed):
            logger.warning("NaN input detected in LongitudinalPID – applying brake.")
            return 0.0, 1.0

        error = target_speed - current_speed

        # ── Proportional term ────────────────────────────────────────────
        P = self.K_P * error

        # ── Integral term with anti-windup ───────────────────────────────
        self._error_integral += error * self.dt
        self._error_integral = max(
            min(self._error_integral, self.integral_max), -self.integral_max
        )
        I = self.K_I * self._error_integral

        # ── Derivative term with EMA smoothing ───────────────────────────
        raw_derivative = _safe_div(error - self._previous_error, self.dt)
        self._error_derivative = (
            self.ema_alpha * raw_derivative
            + (1.0 - self.ema_alpha) * self._error_derivative
        )
        D = self.K_D * self._error_derivative

        self._previous_error = error

        # ── Store for debugging ──────────────────────────────────────────
        self._last_P = P
        self._last_I = I
        self._last_D = D

        # ── Total control effort → throttle / brake ──────────────────────
        control = P + I + D

        if control >= 0.0:
            throttle = min(control, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(control), 1.0)

        logger.debug(
            "LON PID | err=%.3f P=%.3f I=%.3f D=%.3f → thr=%.3f brk=%.3f",
            error, P, I, D, throttle, brake,
        )
        return throttle, brake

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def reset(self):
        """Reset all internal state (call on vehicle respawn)."""
        self._error_integral = 0.0
        self._error_derivative = 0.0
        self._previous_error = 0.0
        self._last_P = 0.0
        self._last_I = 0.0
        self._last_D = 0.0

    def debug_info(self) -> dict:
        """Return a snapshot of the last PID terms for telemetry / plots."""
        return {
            "P": self._last_P,
            "I": self._last_I,
            "D": self._last_D,
            "integral_accum": self._error_integral,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  LATERAL PID — steering control
# ═════════════════════════════════════════════════════════════════════════════
class LateralPID:
    """
    PID controller for lateral (steering) control.

    Computes a steering command from the cross-track error between the
    vehicle's heading and a target waypoint.

    Tuning Parameters (constructor args)
    -------------------------------------
    K_P : float  – Proportional gain.   Start ≈ 1.0–1.5.
    K_I : float  – Integral gain.       Keep very low (≤ 0.01).
    K_D : float  – Derivative gain.     Start ≈ 0.1–0.3.
    dt  : float  – Expected time-step in seconds.

    Advanced Settings (attributes)
    ------------------------------
    integral_max      : float – Anti-windup clamp.
    ema_alpha         : float – Derivative smoothing factor.
    max_steer         : float – Hard steering clamp (CARLA max is 1.0).
    max_steer_rate    : float – Maximum change in steering per step
                                (prevents jerky transitions).
    """

    def __init__(
        self,
        K_P: float = 1.0,
        K_I: float = 0.0,
        K_D: float = 0.1,
        dt: float = 0.05,
    ):
        # ── Gains ────────────────────────────────────────────────────────
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.dt = dt

        # ── Internal state ───────────────────────────────────────────────
        self._error_integral: float = 0.0
        self._error_derivative: float = 0.0
        self._previous_error: float = 0.0
        self._previous_steer: float = 0.0  # for rate limiting

        # ── Anti-windup clamp ────────────────────────────────────────────
        self.integral_max: float = 2.0

        # ── Derivative EMA smoothing ─────────────────────────────────────
        self.ema_alpha: float = 0.4

        # ── Steering clamps ─────────────────────────────────────────────
        self.max_steer: float = 1.0        # CARLA hard limit
        self.max_steer_rate: float = 0.15  # max Δsteer per tick (smooth turns)

        # ── Debug bookkeeping ────────────────────────────────────────────
        self._last_P: float = 0.0
        self._last_I: float = 0.0
        self._last_D: float = 0.0
        self._last_cte: float = 0.0

    # ------------------------------------------------------------------
    #  Main computation
    # ------------------------------------------------------------------
    def run_step(self, target_point, current_transform):
        """
        Compute a steering command from a target waypoint.

        Parameters
        ----------
        target_point      : tuple (x, y)
            The waypoint the vehicle should steer towards.
        current_transform : tuple (x, y, yaw_rad)
            Vehicle position and heading.  **yaw must be in radians.**

        Returns
        -------
        steering : float ∈ [-max_steer, +max_steer]
            Negative = steer left, Positive = steer right.

        Cross-Track Error Convention
        ----------------------------
        The CTE is the signed perpendicular distance from the vehicle's
        heading line to the target point.  It is computed via the 2-D cross
        product of the heading unit-vector and the vehicle→target vector.
        """
        # ── Safety: reject garbage inputs ────────────────────────────────
        if any(math.isnan(v) for v in (*target_point[:2], *current_transform[:3])):
            logger.warning("NaN input detected in LateralPID – holding steer.")
            return self._previous_steer

        tx, ty = target_point[0], target_point[1]
        cx, cy, cyaw = current_transform[0], current_transform[1], current_transform[2]

        # ── Heading unit-vector ──────────────────────────────────────────
        hx = math.cos(cyaw)
        hy = math.sin(cyaw)

        # ── Vector from vehicle to target ────────────────────────────────
        dx = tx - cx
        dy = ty - cy

        # ── Cross-track error (2-D cross product) ───────────────────────
        #    Positive CTE → target is to the LEFT of the heading
        #    Negative CTE → target is to the RIGHT
        crosstrack_error = (hy * dx) - (hx * dy)
        self._last_cte = crosstrack_error

        # ── Proportional ─────────────────────────────────────────────────
        P = self.K_P * crosstrack_error

        # ── Integral with anti-windup ────────────────────────────────────
        self._error_integral += crosstrack_error * self.dt
        self._error_integral = max(
            min(self._error_integral, self.integral_max), -self.integral_max
        )
        I = self.K_I * self._error_integral

        # ── Derivative with EMA smoothing ────────────────────────────────
        raw_deriv = _safe_div(crosstrack_error - self._previous_error, self.dt)
        self._error_derivative = (
            self.ema_alpha * raw_deriv
            + (1.0 - self.ema_alpha) * self._error_derivative
        )
        D = self.K_D * self._error_derivative

        self._previous_error = crosstrack_error

        # ── Store for debug ──────────────────────────────────────────────
        self._last_P = P
        self._last_I = I
        self._last_D = D

        # ── Raw steering command ─────────────────────────────────────────
        raw_steer = P + I + D

        # ── Rate limiter (smooth turning) ────────────────────────────────
        #    Limits how fast steering can change between ticks.
        delta = raw_steer - self._previous_steer
        if abs(delta) > self.max_steer_rate:
            raw_steer = self._previous_steer + math.copysign(self.max_steer_rate, delta)

        # ── Hard clamp ───────────────────────────────────────────────────
        steering = max(min(raw_steer, self.max_steer), -self.max_steer)
        self._previous_steer = steering

        logger.debug(
            "LAT PID | cte=%.3f P=%.3f I=%.3f D=%.3f → steer=%.3f",
            crosstrack_error, P, I, D, steering,
        )
        return steering

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def reset(self):
        """Reset all internal state (call on vehicle respawn)."""
        self._error_integral = 0.0
        self._error_derivative = 0.0
        self._previous_error = 0.0
        self._previous_steer = 0.0
        self._last_P = 0.0
        self._last_I = 0.0
        self._last_D = 0.0
        self._last_cte = 0.0

    def debug_info(self) -> dict:
        """Return a snapshot of the last PID terms for telemetry / plots."""
        return {
            "P": self._last_P,
            "I": self._last_I,
            "D": self._last_D,
            "CTE": self._last_cte,
            "integral_accum": self._error_integral,
            "steer_output": self._previous_steer,
        }
