"""
=============================================================================
Finite State Machine (FSM) for Autonomous Decision Making
=============================================================================

This module implements a priority-based FSM that evaluates perception data
every tick and selects the appropriate driving behaviour.

State Hierarchy (highest to lowest priority)
---------------------------------------------
1. EMERGENCY_STOP  – Imminent collision or red traffic light
2. OBSTACLE_AVOIDANCE – Obstacle on the planned path, attempt lateral dodge
3. BRAKE           – Vehicle / obstacle within safe-follow distance
4. TURN            – Upcoming sharp curve detected from waypoints
5. LANE_FOLLOW     – Normal driving along the planned route
6. CRUISE          – Open road, no waypoints constraining behaviour

The FSM is intentionally lightweight (no deep learning) and relies on
simple arithmetic comparisons for real-time performance.

Author : Controls Team
Version: 1.0.0
"""

from enum import Enum, auto
import math
import logging
import time

# ── Module-level logger ─────────────────────────────────────────────────────
logger = logging.getLogger("controls.fsm")


# ═════════════════════════════════════════════════════════════════════════════
#  STATE DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════
class VehicleState(Enum):
    """All possible behavioural states the vehicle can be in."""
    CRUISE             = auto()  # Open road, no special behaviour
    LANE_FOLLOW        = auto()  # Following waypoints on the lane
    TURN               = auto()  # Upcoming sharp curve — reduce speed
    BRAKE              = auto()  # Object ahead within follow distance
    EMERGENCY_STOP     = auto()  # Imminent collision — full brake
    OBSTACLE_AVOIDANCE = auto()  # Lateral manoeuvre to avoid obstacle


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER: estimate curvature from three waypoints
# ═════════════════════════════════════════════════════════════════════════════
def _estimate_curvature(p0, p1, p2):
    """
    Menger curvature of three 2-D points.

    Returns 1/radius.  A high value means a tight curve.
    Returns 0.0 when the points are (nearly) collinear.

    Uses the formula:
        κ = 4·area(triangle) / (|AB|·|BC|·|CA|)
    """
    ax, ay = p0
    bx, by = p1
    cx, cy = p2

    # Twice the signed area of the triangle
    area2 = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))

    # Side lengths
    ab = math.hypot(bx - ax, by - ay)
    bc = math.hypot(cx - bx, cy - by)
    ca = math.hypot(ax - cx, ay - cy)

    denom = ab * bc * ca
    if denom < 1e-9:
        return 0.0  # collinear or coincident points

    return area2 / denom


# ═════════════════════════════════════════════════════════════════════════════
#  FSM DECISION MAKER
# ═════════════════════════════════════════════════════════════════════════════
class FSMDecisionMaker:
    """
    Lightweight rule-based decision maker.

    Constructor Parameters
    ----------------------
    target_cruise_speed   : float – Desired cruising speed (m/s).
    turn_speed            : float – Max speed in a detected curve (m/s).
    emergency_brake_dist  : float – Distance (m) below which we do full brake.
    safe_follow_dist      : float – Distance (m) at which we start slowing.
    avoidance_lateral_m   : float – If obstacle is within this many metres of
                                    the planned path, enter avoidance.
    curvature_threshold   : float – Menger curvature above which we slow down.
    """

    def __init__(
        self,
        target_cruise_speed: float = 10.0,
        turn_speed: float = 4.0,
        emergency_brake_dist: float = 5.0,
        safe_follow_dist: float = 15.0,
        avoidance_lateral_m: float = 3.0,
        curvature_threshold: float = 0.05,
    ):
        # ── Speeds (m/s) ────────────────────────────────────────────────
        self.target_cruise_speed = target_cruise_speed
        self.turn_speed = turn_speed

        # ── Distance thresholds (metres) ────────────────────────────────
        self.emergency_brake_dist = emergency_brake_dist
        self.safe_follow_dist = safe_follow_dist
        self.avoidance_lateral_m = avoidance_lateral_m

        # ── Curvature threshold ─────────────────────────────────────────
        self.curvature_threshold = curvature_threshold

        # ── Runtime state ───────────────────────────────────────────────
        self.current_state: VehicleState = VehicleState.CRUISE
        self._state_enter_time: float = time.time()

        # ── Logging ─────────────────────────────────────────────────────
        logger.info(
            "FSM initialised | cruise=%.1f m/s  turn=%.1f m/s  "
            "e-brake=%.1fm  follow=%.1fm  curv_thresh=%.4f",
            target_cruise_speed, turn_speed,
            emergency_brake_dist, safe_follow_dist, curvature_threshold,
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def evaluate(self, perception_data: dict, current_speed: float, waypoints: list):
        """
        Evaluate sensor / perception information and choose the next state.

        Parameters
        ----------
        perception_data : dict | None
            Expected keys (all optional):
                'obstacles'      – list of dicts, each with at least
                                   'distance' (m) and optionally
                                   'lateral_offset' (m, signed).
                'traffic_light'  – str, one of 'RED', 'YELLOW', 'GREEN',
                                   or None.
                'speed_limit'    – float, posted speed limit in m/s (opt).
        current_speed : float
            Vehicle speed in m/s.
        waypoints : list[(float, float)]
            Upcoming waypoints as (x, y) tuples.

        Returns
        -------
        (state, target_speed, target_waypoint) :
            tuple[VehicleState, float, tuple|None]
        """
        # ── Defaults ─────────────────────────────────────────────────────
        target_speed = self.target_cruise_speed
        target_waypoint = waypoints[0] if waypoints else None

        # ── Respect speed limits if provided ─────────────────────────────
        if perception_data and perception_data.get("speed_limit") is not None:
            target_speed = min(target_speed, perception_data["speed_limit"])

        # ==============================================================
        # PRIORITY 1 — EMERGENCY STOP  (imminent collision / red light)
        # ==============================================================
        closest_dist = self._closest_obstacle_distance(perception_data)

        is_red = (
            perception_data.get("traffic_light") in ("RED", "YELLOW")
            if perception_data
            else False
        )

        if closest_dist < self.emergency_brake_dist or is_red:
            self._transition(VehicleState.EMERGENCY_STOP)
            logger.debug("→ EMERGENCY_STOP  obs=%.2fm  red=%s", closest_dist, is_red)
            return VehicleState.EMERGENCY_STOP, 0.0, target_waypoint

        # ==============================================================
        # PRIORITY 2 — OBSTACLE AVOIDANCE  (lateral dodge)
        # ==============================================================
        if self._should_avoid(perception_data):
            self._transition(VehicleState.OBSTACLE_AVOIDANCE)
            # Slow down while avoiding
            avoid_speed = min(target_speed, self.turn_speed)
            logger.debug("→ OBSTACLE_AVOIDANCE  speed=%.2f", avoid_speed)
            return VehicleState.OBSTACLE_AVOIDANCE, avoid_speed, target_waypoint

        # ==============================================================
        # PRIORITY 3 — BRAKE  (car-following / obstacle in path)
        # ==============================================================
        if closest_dist < self.safe_follow_dist:
            self._transition(VehicleState.BRAKE)
            # Linearly scale speed between emergency and safe distances
            ratio = max(
                (closest_dist - self.emergency_brake_dist)
                / (self.safe_follow_dist - self.emergency_brake_dist),
                0.0,
            )
            target_speed = target_speed * ratio
            logger.debug(
                "→ BRAKE  obs=%.2fm  ratio=%.2f  tgt_spd=%.2f",
                closest_dist, ratio, target_speed,
            )
            return VehicleState.BRAKE, target_speed, target_waypoint

        # ==============================================================
        # PRIORITY 4 — TURN  (sharp curve ahead)
        # ==============================================================
        if waypoints and len(waypoints) >= 3:
            curvature = _estimate_curvature(
                waypoints[0],
                waypoints[min(1, len(waypoints) - 1)],
                waypoints[min(2, len(waypoints) - 1)],
            )
            if curvature > self.curvature_threshold:
                self._transition(VehicleState.TURN)
                # Dynamic speed: tighter curve → slower
                #   curvature ≈ 1/radius, so radius = 1/curvature
                radius = 1.0 / max(curvature, 1e-6)
                # Simple model: v = sqrt(a_lat_max * radius), cap at cruise
                max_lateral_accel = 3.0  # m/s² — comfortable cornering
                curve_speed = min(
                    math.sqrt(max_lateral_accel * radius),
                    target_speed,
                )
                # Never go below a minimum creep speed
                curve_speed = max(curve_speed, 2.0)
                logger.debug(
                    "→ TURN  κ=%.4f  R=%.1fm  curve_spd=%.2f",
                    curvature, radius, curve_speed,
                )
                return VehicleState.TURN, curve_speed, target_waypoint

        # ==============================================================
        # PRIORITY 5 — LANE FOLLOW  (normal driving with waypoints)
        # ==============================================================
        if waypoints:
            self._transition(VehicleState.LANE_FOLLOW)
            return VehicleState.LANE_FOLLOW, target_speed, target_waypoint

        # ==============================================================
        # PRIORITY 6 — CRUISE  (no waypoints, open road)
        # ==============================================================
        self._transition(VehicleState.CRUISE)
        return VehicleState.CRUISE, target_speed, None

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _closest_obstacle_distance(self, perception_data: dict) -> float:
        """Return distance to the closest obstacle, or inf."""
        if not perception_data or not perception_data.get("obstacles"):
            return float("inf")
        return min(
            obs.get("distance", float("inf"))
            for obs in perception_data["obstacles"]
        )

    def _should_avoid(self, perception_data: dict) -> bool:
        """
        Return True if any obstacle is dangerously close laterally
        (i.e. blocking the planned lane but beyond emergency-brake distance).
        """
        if not perception_data or not perception_data.get("obstacles"):
            return False
        for obs in perception_data["obstacles"]:
            lat = abs(obs.get("lateral_offset", float("inf")))
            dist = obs.get("distance", float("inf"))
            if lat < self.avoidance_lateral_m and dist < self.safe_follow_dist:
                return True
        return False

    def _transition(self, new_state: VehicleState):
        """Transition to a new state and log the change."""
        if new_state != self.current_state:
            elapsed = time.time() - self._state_enter_time
            logger.info(
                "FSM %s → %s  (was in %s for %.2fs)",
                self.current_state.name,
                new_state.name,
                self.current_state.name,
                elapsed,
            )
            self.current_state = new_state
            self._state_enter_time = time.time()

    # ------------------------------------------------------------------
    #  Debug
    # ------------------------------------------------------------------
    def debug_info(self) -> dict:
        """Return FSM telemetry snapshot."""
        return {
            "state": self.current_state.name,
            "time_in_state": time.time() - self._state_enter_time,
        }
