# Autonomous Vehicle Controls Guide

> **Rule-based PID + Finite State Machine control system.**
> No deep learning required — pure arithmetic optimised for real-time CARLA.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [PID Tuning Guide](#pid-tuning-guide)
4. [FSM Workflow](#fsm-workflow)
5. [Optimization Strategies](#optimization-strategies)
6. [CARLA Integration Steps](#carla-integration-steps)
7. [Debugging & Telemetry](#debugging--telemetry)

---

## Architecture Overview

```
  Perception Layer                  Controls Layer
 ┌────────────────┐      ┌───────────────────────────────────┐
 │ Lane Detection │      │         VehicleController         │
 │ Object Detect. │ ───► │  ┌─────────┐   ┌──────────────┐  │
 │ Traffic Lights │      │  │   FSM   │──►│ Target Speed │  │───► carla.VehicleControl
 └────────────────┘      │  └─────────┘   │ Target WP    │  │     (throttle, steer, brake)
                         │                └──────┬───────┘  │
  CARLA Simulator        │        ┌──────────────┴───────┐  │
 ┌────────────────┐      │  ┌─────┴─────┐  ┌────────────┐  │
 │ Vehicle State  │ ───► │  │ LON PID   │  │ LAT PID    │  │
 │ (pos, speed)   │      │  │ (thr/brk) │  │ (steering) │  │
 └────────────────┘      │  └───────────┘  └────────────┘  │
                         └───────────────────────────────────┘
```

The **VehicleController** is the single entry-point.  Each tick it:
1. Selects a **lookahead waypoint** using distance-based search.
2. Queries the **FSM** for the behavioural state and target speed.
3. Runs the **Longitudinal PID** → throttle / brake.
4. Runs the **Lateral PID** → steering.
5. Applies safety overrides (emergency stop, avoidance bias).

---

## Project Structure

```
controls/
├── __init__.py              # Public API exports
├── pid_controller.py        # LongitudinalPID, LateralPID
├── fsm_decision_maker.py    # VehicleState enum, FSMDecisionMaker
├── vehicle_controller.py    # VehicleController (master class)
└── CONTROLS_GUIDE.md        # ← You are here
```

---

## PID Tuning Guide

### General Method

1. **Set K_I = 0, K_D = 0.**
2. **Increase K_P** until the system responds quickly but starts oscillating.
3. **Increase K_D** to dampen the oscillation.
4. **Add a small K_I** only if there is persistent steady-state error.

### Longitudinal PID (Speed Control)

| Gain | Default | Effect |
|------|---------|--------|
| K_P  | 1.0     | Responsiveness. ↑ = faster reach, but can oscillate |
| K_I  | 0.05    | Eliminates offset (e.g., uphill). Keep small |
| K_D  | 0.01    | Dampens overshoot. ↑ = smoother, but sluggish |

**Anti-windup**: `integral_max = 5.0`.  The integral accumulator is clamped
to `[-5, +5]` to prevent build-up when the car is stopped for long periods.

**Derivative smoothing**: EMA with `alpha = 0.4`.  Lower alpha = heavier
smoothing (less sensor-noise jitter, but slower reaction).

### Lateral PID (Steering)

| Gain | Default | Effect |
|------|---------|--------|
| K_P  | 1.0     | Aggressiveness of turn-in. Too high → snaking |
| K_I  | 0.0     | Almost never needed for steering |
| K_D  | 0.1     | Anticipates error, smooths return to center |

**Steering rate limiter**: `max_steer_rate = 0.15` per tick.
Prevents the steering from jumping instantly, giving a smooth turning feel.

**Hard clamp**: `max_steer = 1.0` (CARLA's physical limit).

---

## FSM Workflow

### States (Priority Order)

```
┌─────────────────────────────────────────────────────────────────┐
│ Priority   State                Trigger                        │
├─────────────────────────────────────────────────────────────────┤
│ 1 (HIGH)   EMERGENCY_STOP       obstacle < 5 m  OR  RED light │
│ 2          OBSTACLE_AVOIDANCE   obstacle lateral < 3 m         │
│ 3          BRAKE                obstacle < 15 m (follow zone)  │
│ 4          TURN                 curvature > threshold          │
│ 5          LANE_FOLLOW          waypoints available            │
│ 6 (LOW)    CRUISE               no waypoints                   │
└─────────────────────────────────────────────────────────────────┘
```

### Dynamic Speed Adaptation

| State              | Speed Logic |
|--------------------|-------------|
| EMERGENCY_STOP     | 0 m/s (full brake) |
| OBSTACLE_AVOIDANCE | min(cruise, turn_speed) |
| BRAKE              | cruise × linear_ratio(distance) |
| TURN               | v = √(a_lat_max × R), capped at cruise, min 2 m/s |
| LANE_FOLLOW        | cruise speed (or speed limit if provided) |
| CRUISE             | cruise speed |

### Curve Detection

Uses **Menger curvature** from three consecutive waypoints:
```
κ = 4 × area(triangle) / (|AB| × |BC| × |CA|)
```
When κ exceeds the threshold (default 0.05), the FSM enters TURN and
computes a safe curve speed using:  `v = √(a_lat_max × R)` where `R = 1/κ`.

---

## Optimization Strategies

| Strategy | Implementation |
|----------|----------------|
| No DL inference | Pure arithmetic — runs in < 0.1 ms/tick |
| Tuple unpacking | `cx, cy, cyaw = transform` — no object creation |
| Squared distances | `_distance_sq()` avoids `sqrt` during waypoint search |
| EMA derivative | Single multiply+add instead of ring-buffer filter |
| Early exit | FSM evaluates highest-priority conditions first |
| Minimal allocation | No lists/dicts created inside the hot path |

### Performance Monitoring

`VehicleController` tracks compute time internally and logs average ms/tick
every 200 ticks.  Access any time via:
```python
info = controller.debug_info()
print(info["avg_compute_ms"])  # e.g. 0.042
```

---

## CARLA Integration Steps

### Step 1 — Install

No extra dependencies.  The controls module uses only Python stdlib (`math`,
`enum`, `logging`, `time`).

### Step 2 — Create the Controller

```python
from controls import VehicleController

controller = VehicleController(
    lon_kp=1.0,  lon_ki=0.05,  lon_kd=0.01,
    lat_kp=1.0,  lat_ki=0.0,   lat_kd=0.1,
    dt=0.05,                    # must match CARLA's fixed_delta_seconds
    target_cruise_speed=10.0,   # m/s ≈ 36 km/h
    lookahead_distance=5.0,     # metres
)
```

### Step 3 — Simulation Loop

```python
import carla, math

# Inside the tick loop:
transform = vehicle.get_transform()
velocity  = vehicle.get_velocity()

speed = math.sqrt(velocity.x**2 + velocity.y**2)

transform_tuple = (
    transform.location.x,
    transform.location.y,
    math.radians(transform.rotation.yaw),
)

# Build perception dict from your detection modules
perception_data = {
    "obstacles": [{"distance": 12.0, "lateral_offset": 1.5}],
    "traffic_light": "GREEN",       # or "RED" / "YELLOW"
    "speed_limit": 8.33,            # optional, m/s
}

# Waypoints from CARLA GlobalRoutePlanner or your own planner
local_waypoints = [(wp.transform.location.x, wp.transform.location.y)
                   for wp in route_waypoints[:20]]

throttle, steer, brake = controller.get_control(
    current_transform=transform_tuple,
    current_speed=speed,
    waypoints=local_waypoints,
    perception_data=perception_data,
)

vehicle.apply_control(
    carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
)
```

### Step 4 — Enable Logging (optional)

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# or selectively:
logging.getLogger("controls.pid").setLevel(logging.WARNING)
logging.getLogger("controls.fsm").setLevel(logging.INFO)
logging.getLogger("controls.vehicle").setLevel(logging.DEBUG)
```

---

## Debugging & Telemetry

All three modules expose a `debug_info()` method:

```python
info = controller.debug_info()
# {
#   "fsm":     {"state": "LANE_FOLLOW", "time_in_state": 3.21},
#   "lon_pid": {"P": 0.42, "I": 0.01, "D": -0.003, "integral_accum": 0.12},
#   "lat_pid": {"P": 0.15, "I": 0.0, "D": -0.02, "CTE": 0.15,
#               "integral_accum": 0.0, "steer_output": 0.13},
#   "tick": 1024,
#   "avg_compute_ms": 0.038,
# }
```

Use this data to:
- Plot PID terms over time (diagnose oscillation, windup).
- Monitor FSM state durations (detect stuck states).
- Verify real-time performance (should stay well below 1 ms/tick).
