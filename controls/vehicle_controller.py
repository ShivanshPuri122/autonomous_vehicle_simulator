"""
=============================================================================
Vehicle Controller — Master Integration of FSM + PID
=============================================================================

This is the single entry-point that the CARLA simulation loop should call
every tick.  It:

    1.  Selects the best lookahead waypoint for steering.
    2.  Queries the FSM to determine the driving state and target speed.
    3.  Runs the Longitudinal PID to produce throttle / brake.
    4.  Runs the Lateral PID to produce steering.
    5.  Applies safety overrides (emergency stop, speed clamps).
    6.  Returns (throttle, steer, brake) ready for ``carla.VehicleControl``.

Usage
-----
    from controls import VehicleController

    ctrl = VehicleController(dt=0.05)

    # inside your simulation tick:
    throttle, steer, brake = ctrl.get_control(
        current_transform=(x, y, yaw_rad),
        current_speed=speed_ms,
        waypoints=[(x1,y1), (x2,y2), ...],
        perception_data={'obstacles': [...], 'traffic_light': 'GREEN'},
    )

Author : Controls Team
Version: 1.0.0
"""

import math
import logging
import time

from .pid_controller import LongitudinalPID, LateralPID
from .fsm_decision_maker import FSMDecisionMaker, VehicleState

# ── Module-level logger ─────────────────────────────────────────────────────
logger = logging.getLogger("controls.vehicle")


# ═════════════════════════════════════════════════════════════════════════════
#  WAYPOINT HELPERS  (optimised — pure arithmetic, no numpy)
# ═════════════════════════════════════════════════════════════════════════════
def _distance_sq(ax: float, ay: float, bx: float, by: float) -> float:
    """Squared Euclidean distance (avoids sqrt for comparisons)."""
    dx = bx - ax
    dy = by - ay
    return dx * dx + dy * dy


def _find_lookahead_waypoint(
    cx: float, cy: float, waypoints: list, lookahead: float = 5.0
):
    """
    Find the first waypoint that is at least ``lookahead`` metres ahead
    of the vehicle position (cx, cy).

    Falls back to the farthest waypoint if none exceeds the lookahead.

    Parameters
    ----------
    cx, cy     : float – Vehicle position.
    waypoints  : list[(x, y)] – Ordered upcoming waypoints.
    lookahead  : float – Minimum lookahead distance in metres.

    Returns
    -------
    (wx, wy)   : tuple – Selected target waypoint.
    index      : int   – Index into the waypoints list.
    """
    la_sq = lookahead * lookahead
    for idx, (wx, wy) in enumerate(waypoints):
        if _distance_sq(cx, cy, wx, wy) >= la_sq:
            return (wx, wy), idx
    # Fallback — use the last waypoint
    return waypoints[-1], len(waypoints) - 1


# ═════════════════════════════════════════════════════════════════════════════
#  VEHICLE CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════
class VehicleController:
    """
    Master controller that fuses FSM decisions with PID outputs.

    Constructor Parameters
    ----------------------
    lon_kp / lon_ki / lon_kd : float – Longitudinal PID gains.
    lat_kp / lat_ki / lat_kd : float – Lateral PID gains.
    dt                       : float – Simulation time-step (seconds).
    target_cruise_speed      : float – Desired cruising speed (m/s).
    lookahead_distance       : float – Lateral-PID lookahead (metres).
    """

    def __init__(
        self,
        lon_kp: float = 1.0,
        lon_ki: float = 0.05,
        lon_kd: float = 0.01,
        lat_kp: float = 1.0,
        lat_ki: float = 0.0,
        lat_kd: float = 0.1,
        dt: float = 0.05,
        target_cruise_speed: float = 10.0,
        lookahead_distance: float = 5.0,
    ):
        # ── PID controllers ─────────────────────────────────────────────
        self.lon_pid = LongitudinalPID(K_P=lon_kp, K_I=lon_ki, K_D=lon_kd, dt=dt)
        self.lat_pid = LateralPID(K_P=lat_kp, K_I=lat_ki, K_D=lat_kd, dt=dt)

        # ── FSM ─────────────────────────────────────────────────────────
        self.fsm = FSMDecisionMaker(target_cruise_speed=target_cruise_speed)

        # ── Waypoint navigation ─────────────────────────────────────────
        self.lookahead_distance = lookahead_distance

        # ── Tick counter for performance monitoring ─────────────────────
        self._tick: int = 0
        self._total_compute_ms: float = 0.0

        logger.info(
            "VehicleController created | lon=(%.2f,%.2f,%.2f) "
            "lat=(%.2f,%.2f,%.2f) dt=%.3fs  lookahead=%.1fm",
            lon_kp, lon_ki, lon_kd, lat_kp, lat_ki, lat_kd, dt,
            lookahead_distance,
        )

    # ------------------------------------------------------------------
    #  Main entry-point — call this every CARLA tick
    # ------------------------------------------------------------------
    def get_control(
        self,
        current_transform: tuple,
        current_speed: float,
        waypoints: list,
        perception_data: dict = None,
    ):
        """
        Compute throttle, steer, and brake for the current tick.

        Parameters
        ----------
        current_transform : tuple (x, y, yaw_rad)
            Vehicle position and heading.  **yaw must be in radians.**
        current_speed : float
            Vehicle speed in m/s.
        waypoints : list[(float, float)]
            Ordered list of upcoming waypoints as (x, y) tuples.
        perception_data : dict | None
            See ``FSMDecisionMaker.evaluate`` for expected format.

        Returns
        -------
        (throttle, steer, brake) : tuple[float, float, float]
            throttle ∈ [0, 1], steer ∈ [−1, 1], brake ∈ [0, 1].
        """
        t0 = time.perf_counter()

        # ── Safety: no waypoints → full brake ────────────────────────────
        if not waypoints:
            logger.warning("No waypoints provided — emergency braking.")
            return 0.0, 0.0, 1.0

        cx, cy, cyaw = current_transform

        # ==============================================================
        # STEP 1 — Pick a lookahead waypoint for steering
        # ==============================================================
        target_wp, wp_idx = _find_lookahead_waypoint(
            cx, cy, waypoints, self.lookahead_distance
        )

        # ==============================================================
        # STEP 2 — FSM decision (state + target speed)
        # ==============================================================
        state, target_speed, _ = self.fsm.evaluate(
            perception_data=perception_data,
            current_speed=current_speed,
            waypoints=waypoints,
        )

        # ==============================================================
        # STEP 3 — Longitudinal PID (throttle / brake)
        # ==============================================================
        throttle, brake = self.lon_pid.run_step(target_speed, current_speed)

        # ==============================================================
        # STEP 4 — Lateral PID (steering)
        # ==============================================================
        steer = self.lat_pid.run_step(target_wp, current_transform)

        # ==============================================================
        # STEP 5 — State-specific overrides
        # ==============================================================
        if state == VehicleState.EMERGENCY_STOP:
            throttle = 0.0
            brake = 1.0
            # Keep last steer to avoid snapping wheels straight
            # while braking hard in a curve.

        if state == VehicleState.OBSTACLE_AVOIDANCE:
            # Add a small lateral bias away from the obstacle.
            # (A more advanced implementation could plan a full path.)
            if perception_data and perception_data.get("obstacles"):
                for obs in perception_data["obstacles"]:
                    lat_off = obs.get("lateral_offset", 0.0)
                    if lat_off != 0.0:
                        # Steer away from the obstacle
                        avoidance_bias = -0.3 * math.copysign(1.0, lat_off)
                        steer = max(min(steer + avoidance_bias, 1.0), -1.0)
                        break  # react to the first relevant obstacle

        # ==============================================================
        # STEP 6 — Performance bookkeeping
        # ==============================================================
        self._tick += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._total_compute_ms += elapsed_ms

        if self._tick % 200 == 0:
            avg_ms = self._total_compute_ms / self._tick
            logger.info(
                "Controller tick #%d | avg compute = %.3f ms/tick",
                self._tick, avg_ms,
            )

        logger.debug(
            "CTRL tick=%d state=%-18s tgt_spd=%.2f spd=%.2f "
            "thr=%.3f steer=%.3f brk=%.3f  wp_idx=%d  (%.3fms)",
            self._tick, state.name, target_speed, current_speed,
            throttle, steer, brake, wp_idx, elapsed_ms,
        )

        return throttle, steer, brake

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def reset(self):
        """Reset all internal state (call on vehicle respawn)."""
        self.lon_pid.reset()
        self.lat_pid.reset()
        self._tick = 0
        self._total_compute_ms = 0.0
        logger.info("VehicleController reset.")

    def debug_info(self) -> dict:
        """Aggregate debug snapshot from all sub-modules."""
        return {
            "fsm": self.fsm.debug_info(),
            "lon_pid": self.lon_pid.debug_info(),
            "lat_pid": self.lat_pid.debug_info(),
            "tick": self._tick,
            "avg_compute_ms": (
                self._total_compute_ms / self._tick if self._tick > 0 else 0.0
            ),
        }
