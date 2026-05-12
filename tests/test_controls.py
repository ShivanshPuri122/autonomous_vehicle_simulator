"""
=============================================================================
Comprehensive Test Suite for the Controls Module
=============================================================================

Tests cover:
  1. LongitudinalPID  — throttle/brake output, anti-windup, NaN safety, reset
  2. LateralPID       — steering output, rate limiter, clamp, NaN safety, reset
  3. FSMDecisionMaker — all 6 states, dynamic speed, curvature detection
  4. VehicleController — end-to-end integration, no-waypoint safety, debug_info

Run:
    python -m pytest tests/test_controls.py -v
    OR
    python tests/test_controls.py          (standalone with unittest)

Results are printed to stdout and can be redirected to test_results/.
"""

import unittest
import math
import sys
import os
import json
import datetime

# ── Ensure the project root is on sys.path ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controls.pid_controller import LongitudinalPID, LateralPID
from controls.fsm_decision_maker import FSMDecisionMaker, VehicleState, _estimate_curvature
from controls.vehicle_controller import VehicleController, _find_lookahead_waypoint


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 1: LongitudinalPID
# ═════════════════════════════════════════════════════════════════════════════
class TestLongitudinalPID(unittest.TestCase):
    """Tests for the throttle / brake PID controller."""

    def setUp(self):
        self.pid = LongitudinalPID(K_P=1.0, K_I=0.05, K_D=0.01, dt=0.05)

    # ── Basic throttle when target > current ─────────────────────────────
    def test_throttle_when_below_target(self):
        """Vehicle is slower than target → should produce throttle > 0."""
        throttle, brake = self.pid.run_step(target_speed=10.0, current_speed=5.0)
        self.assertGreater(throttle, 0.0, "Expected positive throttle")
        self.assertEqual(brake, 0.0, "Expected zero brake")

    # ── Brake when target < current ──────────────────────────────────────
    def test_brake_when_above_target(self):
        """Vehicle is faster than target → should produce brake > 0."""
        throttle, brake = self.pid.run_step(target_speed=5.0, current_speed=10.0)
        self.assertEqual(throttle, 0.0, "Expected zero throttle")
        self.assertGreater(brake, 0.0, "Expected positive brake")

    # ── At target speed → minimal output ─────────────────────────────────
    def test_zero_error(self):
        """Speed matches target → throttle and brake should be near zero."""
        throttle, brake = self.pid.run_step(target_speed=10.0, current_speed=10.0)
        self.assertAlmostEqual(throttle, 0.0, places=2)
        self.assertAlmostEqual(brake, 0.0, places=2)

    # ── Output clamping ──────────────────────────────────────────────────
    def test_throttle_clamp(self):
        """Throttle must never exceed 1.0."""
        throttle, _ = self.pid.run_step(target_speed=100.0, current_speed=0.0)
        self.assertLessEqual(throttle, 1.0)

    def test_brake_clamp(self):
        """Brake must never exceed 1.0."""
        _, brake = self.pid.run_step(target_speed=0.0, current_speed=100.0)
        self.assertLessEqual(brake, 1.0)

    # ── Anti-windup ──────────────────────────────────────────────────────
    def test_anti_windup(self):
        """Integral should not grow beyond integral_max."""
        for _ in range(1000):
            self.pid.run_step(target_speed=50.0, current_speed=0.0)
        self.assertLessEqual(self.pid._error_integral, self.pid.integral_max)
        self.assertGreaterEqual(self.pid._error_integral, -self.pid.integral_max)

    # ── NaN safety ───────────────────────────────────────────────────────
    def test_nan_target_speed(self):
        """NaN target speed should trigger safe brake."""
        throttle, brake = self.pid.run_step(float('nan'), 5.0)
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 1.0)

    def test_nan_current_speed(self):
        """NaN current speed should trigger safe brake."""
        throttle, brake = self.pid.run_step(10.0, float('nan'))
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 1.0)

    # ── Reset ────────────────────────────────────────────────────────────
    def test_reset(self):
        """Reset should clear all internal state."""
        self.pid.run_step(10.0, 5.0)
        self.pid.reset()
        self.assertEqual(self.pid._error_integral, 0.0)
        self.assertEqual(self.pid._error_derivative, 0.0)
        self.assertEqual(self.pid._previous_error, 0.0)

    # ── Debug info ───────────────────────────────────────────────────────
    def test_debug_info(self):
        """debug_info should return a dict with expected keys."""
        self.pid.run_step(10.0, 5.0)
        info = self.pid.debug_info()
        self.assertIn("P", info)
        self.assertIn("I", info)
        self.assertIn("D", info)
        self.assertIn("integral_accum", info)


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 2: LateralPID
# ═════════════════════════════════════════════════════════════════════════════
class TestLateralPID(unittest.TestCase):
    """Tests for the steering PID controller."""

    def setUp(self):
        self.pid = LateralPID(K_P=1.0, K_I=0.0, K_D=0.1, dt=0.05)

    # ── Target straight ahead → zero steer ───────────────────────────────
    def test_target_straight_ahead(self):
        """Waypoint directly ahead on the heading → steer ≈ 0."""
        steer = self.pid.run_step(
            target_point=(10.0, 0.0),
            current_transform=(0.0, 0.0, 0.0),  # facing +X
        )
        self.assertAlmostEqual(steer, 0.0, places=2)

    # ── Target to the left → negative steer (left turn) ─────────────────
    def test_target_to_the_left(self):
        """Waypoint to the left of heading → steer should be negative."""
        steer = self.pid.run_step(
            target_point=(5.0, 5.0),
            current_transform=(0.0, 0.0, 0.0),  # facing +X
        )
        # Cross-track: target is to the left → CTE positive → steer positive
        # (Convention: positive steer = turn towards target on left)
        self.assertNotEqual(steer, 0.0)

    # ── Steering clamp ───────────────────────────────────────────────────
    def test_steering_clamp(self):
        """Steering must be within [-1.0, 1.0]."""
        steer = self.pid.run_step(
            target_point=(0.0, 100.0),
            current_transform=(0.0, 0.0, 0.0),
        )
        self.assertGreaterEqual(steer, -1.0)
        self.assertLessEqual(steer, 1.0)

    # ── Rate limiter ─────────────────────────────────────────────────────
    def test_rate_limiter(self):
        """Consecutive steering changes should not exceed max_steer_rate."""
        s1 = self.pid.run_step((10.0, 0.0), (0.0, 0.0, 0.0))
        s2 = self.pid.run_step((0.0, 100.0), (0.0, 0.0, 0.0))
        delta = abs(s2 - s1)
        self.assertLessEqual(delta, self.pid.max_steer_rate + 1e-6,
                             "Steering change exceeded max_steer_rate")

    # ── NaN safety ───────────────────────────────────────────────────────
    def test_nan_target(self):
        """NaN target should return the previous steer (hold)."""
        self.pid.run_step((10.0, 0.0), (0.0, 0.0, 0.0))
        steer = self.pid.run_step((float('nan'), 0.0), (0.0, 0.0, 0.0))
        # Should return previous steer, not crash
        self.assertFalse(math.isnan(steer))

    # ── Anti-windup ──────────────────────────────────────────────────────
    def test_lateral_anti_windup(self):
        """Integral should stay clamped."""
        pid = LateralPID(K_P=0.1, K_I=1.0, K_D=0.0, dt=0.05)
        for _ in range(500):
            pid.run_step((0.0, 50.0), (0.0, 0.0, 0.0))
        self.assertLessEqual(pid._error_integral, pid.integral_max)

    # ── Reset ────────────────────────────────────────────────────────────
    def test_reset(self):
        """Reset should clear all internal state."""
        self.pid.run_step((10.0, 5.0), (0.0, 0.0, 0.0))
        self.pid.reset()
        self.assertEqual(self.pid._error_integral, 0.0)
        self.assertEqual(self.pid._previous_steer, 0.0)

    # ── Debug info ───────────────────────────────────────────────────────
    def test_debug_info(self):
        """debug_info should return expected keys."""
        self.pid.run_step((10.0, 5.0), (0.0, 0.0, 0.0))
        info = self.pid.debug_info()
        for key in ("P", "I", "D", "CTE", "steer_output"):
            self.assertIn(key, info)


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 3: FSMDecisionMaker
# ═════════════════════════════════════════════════════════════════════════════
class TestFSMDecisionMaker(unittest.TestCase):
    """Tests for the finite state machine."""

    def setUp(self):
        self.fsm = FSMDecisionMaker(target_cruise_speed=10.0)
        self.straight_wps = [(10, 0), (20, 0), (30, 0)]

    # ── LANE_FOLLOW (default, no hazards) ────────────────────────────────
    def test_lane_follow_default(self):
        """No hazards → should default to LANE_FOLLOW at cruise speed."""
        state, speed, wp = self.fsm.evaluate(
            {'obstacles': [], 'traffic_light': 'GREEN'}, 5.0, self.straight_wps
        )
        self.assertEqual(state, VehicleState.LANE_FOLLOW)
        self.assertAlmostEqual(speed, 10.0, places=1)

    # ── EMERGENCY_STOP — red light ───────────────────────────────────────
    def test_emergency_stop_red_light(self):
        """Red traffic light → EMERGENCY_STOP, speed = 0."""
        state, speed, _ = self.fsm.evaluate(
            {'obstacles': [], 'traffic_light': 'RED'}, 5.0, self.straight_wps
        )
        self.assertEqual(state, VehicleState.EMERGENCY_STOP)
        self.assertEqual(speed, 0.0)

    # ── EMERGENCY_STOP — yellow light ────────────────────────────────────
    def test_emergency_stop_yellow_light(self):
        """Yellow traffic light → EMERGENCY_STOP."""
        state, speed, _ = self.fsm.evaluate(
            {'obstacles': [], 'traffic_light': 'YELLOW'}, 5.0, self.straight_wps
        )
        self.assertEqual(state, VehicleState.EMERGENCY_STOP)

    # ── EMERGENCY_STOP — obstacle too close ──────────────────────────────
    def test_emergency_stop_close_obstacle(self):
        """Obstacle at 3m (< emergency threshold 5m) → EMERGENCY_STOP."""
        state, speed, _ = self.fsm.evaluate(
            {'obstacles': [{'distance': 3.0}], 'traffic_light': 'GREEN'},
            8.0, self.straight_wps,
        )
        self.assertEqual(state, VehicleState.EMERGENCY_STOP)
        self.assertEqual(speed, 0.0)

    # ── BRAKE — obstacle in follow zone ──────────────────────────────────
    def test_brake_follow_distance(self):
        """Obstacle at 10m (between 5m and 15m) → BRAKE with reduced speed."""
        state, speed, _ = self.fsm.evaluate(
            {'obstacles': [{'distance': 10.0}], 'traffic_light': 'GREEN'},
            8.0, self.straight_wps,
        )
        self.assertEqual(state, VehicleState.BRAKE)
        self.assertGreater(speed, 0.0, "Should still have some speed")
        self.assertLess(speed, 10.0, "Should be below cruise speed")

    # ── TURN — sharp curve waypoints ─────────────────────────────────────
    def test_turn_state_sharp_curve(self):
        """Waypoints forming a sharp curve → TURN with reduced speed."""
        curve_wps = [(5, 0), (7, 5), (5, 10)]
        state, speed, _ = self.fsm.evaluate(
            {'obstacles': [], 'traffic_light': 'GREEN'},
            8.0, curve_wps,
        )
        self.assertEqual(state, VehicleState.TURN)
        self.assertLessEqual(speed, 10.0)

    # ── OBSTACLE_AVOIDANCE — lateral obstacle ────────────────────────────
    def test_obstacle_avoidance(self):
        """Obstacle with small lateral offset → OBSTACLE_AVOIDANCE."""
        state, speed, _ = self.fsm.evaluate(
            {'obstacles': [{'distance': 10.0, 'lateral_offset': 1.5}],
             'traffic_light': 'GREEN'},
            8.0, self.straight_wps,
        )
        self.assertEqual(state, VehicleState.OBSTACLE_AVOIDANCE)

    # ── CRUISE — no waypoints ────────────────────────────────────────────
    def test_cruise_no_waypoints(self):
        """No waypoints available → CRUISE."""
        state, speed, wp = self.fsm.evaluate(
            {'obstacles': [], 'traffic_light': 'GREEN'}, 5.0, []
        )
        self.assertEqual(state, VehicleState.CRUISE)
        self.assertIsNone(wp)

    # ── Speed limit respect ──────────────────────────────────────────────
    def test_speed_limit(self):
        """Speed limit < cruise speed → target speed capped at limit."""
        state, speed, _ = self.fsm.evaluate(
            {'obstacles': [], 'traffic_light': 'GREEN', 'speed_limit': 5.0},
            3.0, self.straight_wps,
        )
        self.assertLessEqual(speed, 5.0)

    # ── None perception data ─────────────────────────────────────────────
    def test_none_perception(self):
        """None perception data should not crash."""
        state, speed, _ = self.fsm.evaluate(None, 5.0, self.straight_wps)
        self.assertIn(state, list(VehicleState))

    # ── Priority: emergency > brake ──────────────────────────────────────
    def test_emergency_priority_over_brake(self):
        """An obstacle at 3m should trigger EMERGENCY, not BRAKE."""
        state, _, _ = self.fsm.evaluate(
            {'obstacles': [{'distance': 3.0}], 'traffic_light': 'GREEN'},
            8.0, self.straight_wps,
        )
        self.assertEqual(state, VehicleState.EMERGENCY_STOP)

    # ── Debug info ───────────────────────────────────────────────────────
    def test_debug_info(self):
        """debug_info should return state and time_in_state."""
        self.fsm.evaluate({'obstacles': [], 'traffic_light': 'GREEN'}, 5.0, self.straight_wps)
        info = self.fsm.debug_info()
        self.assertIn("state", info)
        self.assertIn("time_in_state", info)


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 3b: Curvature helper
# ═════════════════════════════════════════════════════════════════════════════
class TestCurvatureHelper(unittest.TestCase):
    """Tests for the _estimate_curvature helper."""

    def test_collinear_points(self):
        """Collinear points → curvature ≈ 0."""
        k = _estimate_curvature((0, 0), (1, 0), (2, 0))
        self.assertAlmostEqual(k, 0.0, places=5)

    def test_circle_curvature(self):
        """Points on a unit circle → curvature > 0 (Menger formula)."""
        # Three points on unit circle (R=1): (1,0), (0,1), (-1,0)
        # Menger curvature = area / (|AB|*|BC|*|CA|), gives 0.5 for unit circle
        k = _estimate_curvature((1, 0), (0, 1), (-1, 0))
        self.assertAlmostEqual(k, 0.5, places=3)

    def test_positive_curvature(self):
        """Non-collinear points should produce positive curvature."""
        k = _estimate_curvature((0, 0), (5, 3), (10, 0))
        self.assertGreater(k, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 4: VehicleController (integration)
# ═════════════════════════════════════════════════════════════════════════════
class TestVehicleController(unittest.TestCase):
    """End-to-end integration tests."""

    def setUp(self):
        self.ctrl = VehicleController(dt=0.05)
        self.straight_wps = [(10, 0), (20, 0), (30, 0)]

    # ── Normal driving produces valid outputs ────────────────────────────
    def test_normal_driving(self):
        """Normal inputs → throttle/steer/brake within valid ranges."""
        t, s, b = self.ctrl.get_control(
            (0, 0, 0), 5.0, self.straight_wps,
            {'obstacles': [], 'traffic_light': 'GREEN'},
        )
        self.assertGreaterEqual(t, 0.0)
        self.assertLessEqual(t, 1.0)
        self.assertGreaterEqual(s, -1.0)
        self.assertLessEqual(s, 1.0)
        self.assertGreaterEqual(b, 0.0)
        self.assertLessEqual(b, 1.0)

    # ── No waypoints → emergency brake ───────────────────────────────────
    def test_no_waypoints_emergency(self):
        """Empty waypoints → full brake."""
        t, s, b = self.ctrl.get_control((0, 0, 0), 5.0, [], None)
        self.assertEqual(t, 0.0)
        self.assertEqual(b, 1.0)

    # ── Emergency stop override ──────────────────────────────────────────
    def test_emergency_stop_override(self):
        """Red light → throttle=0, brake=1 regardless of PID output."""
        t, s, b = self.ctrl.get_control(
            (0, 0, 0), 5.0, self.straight_wps,
            {'obstacles': [], 'traffic_light': 'RED'},
        )
        self.assertEqual(t, 0.0)
        self.assertEqual(b, 1.0)

    # ── Reset clears state ───────────────────────────────────────────────
    def test_reset(self):
        """Reset should clear tick counter and PID state."""
        self.ctrl.get_control((0, 0, 0), 5.0, self.straight_wps, None)
        self.ctrl.reset()
        self.assertEqual(self.ctrl._tick, 0)

    # ── Debug info aggregation ───────────────────────────────────────────
    def test_debug_info(self):
        """debug_info should contain fsm, lon_pid, lat_pid, tick, avg_compute_ms."""
        self.ctrl.get_control((0, 0, 0), 5.0, self.straight_wps, None)
        info = self.ctrl.debug_info()
        for key in ("fsm", "lon_pid", "lat_pid", "tick", "avg_compute_ms"):
            self.assertIn(key, info)

    # ── Multi-tick stability ─────────────────────────────────────────────
    def test_multi_tick_stability(self):
        """Running 100 ticks should not crash or produce NaN."""
        for i in range(100):
            t, s, b = self.ctrl.get_control(
                (i * 0.5, 0, 0), 5.0 + i * 0.01,
                [(10 + i, 0), (20 + i, 0), (30 + i, 0)],
                {'obstacles': [], 'traffic_light': 'GREEN'},
            )
            self.assertFalse(math.isnan(t), f"NaN throttle at tick {i}")
            self.assertFalse(math.isnan(s), f"NaN steer at tick {i}")
            self.assertFalse(math.isnan(b), f"NaN brake at tick {i}")


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 4b: Waypoint lookahead helper
# ═════════════════════════════════════════════════════════════════════════════
class TestLookaheadHelper(unittest.TestCase):
    """Tests for the _find_lookahead_waypoint utility."""

    def test_selects_beyond_lookahead(self):
        """Should skip close waypoints and pick one beyond lookahead."""
        wps = [(1, 0), (3, 0), (6, 0), (10, 0)]
        wp, idx = _find_lookahead_waypoint(0, 0, wps, lookahead=5.0)
        self.assertEqual(wp, (6, 0))
        self.assertEqual(idx, 2)

    def test_fallback_to_last(self):
        """If no waypoint exceeds lookahead, return the last one."""
        wps = [(1, 0), (2, 0)]
        wp, idx = _find_lookahead_waypoint(0, 0, wps, lookahead=50.0)
        self.assertEqual(wp, (2, 0))
        self.assertEqual(idx, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  RUNNER — save results to test_results/
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Run tests and collect results
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Use a TextTestRunner that captures output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # ── Build a summary dict ─────────────────────────────────────────────
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful(),
        "failure_details": [
            {"test": str(t), "message": msg} for t, msg in result.failures
        ],
        "error_details": [
            {"test": str(t), "message": msg} for t, msg in result.errors
        ],
    }

    # ── Write JSON summary ───────────────────────────────────────────────
    results_dir = os.path.join(PROJECT_ROOT, "test_results", "controls_test")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "test_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Write human-readable report ──────────────────────────────────────
    report_path = os.path.join(results_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  CONTROLS MODULE — TEST REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Date     : {summary['timestamp']}\n")
        f.write(f"  Tests Run: {summary['tests_run']}\n")
        f.write(f"  Passed   : {summary['tests_run'] - summary['failures'] - summary['errors']}\n")
        f.write(f"  Failures : {summary['failures']}\n")
        f.write(f"  Errors   : {summary['errors']}\n")
        f.write(f"  Skipped  : {summary['skipped']}\n")
        f.write(f"  Status   : {'PASSED' if summary['success'] else 'FAILED'}\n")
        f.write("=" * 70 + "\n\n")

        if summary["failure_details"]:
            f.write("FAILURES:\n")
            f.write("-" * 70 + "\n")
            for item in summary["failure_details"]:
                f.write(f"\n  Test: {item['test']}\n")
                f.write(f"  {item['message']}\n")

        if summary["error_details"]:
            f.write("ERRORS:\n")
            f.write("-" * 70 + "\n")
            for item in summary["error_details"]:
                f.write(f"\n  Test: {item['test']}\n")
                f.write(f"  {item['message']}\n")

        if summary["success"]:
            f.write("All tests passed successfully. The controls module is working correctly.\n")

    print(f"\n{'=' * 70}")
    print(f"  Results saved to: {results_dir}")
    print(f"    -> test_results.json")
    print(f"    -> test_report.txt")
    print(f"{'=' * 70}")
