"""
controls — Rule-based PID + FSM autonomous driving package.

Quick start
-----------
    from controls import VehicleController

    ctrl = VehicleController(dt=0.05)
    throttle, steer, brake = ctrl.get_control(
        current_transform=(x, y, yaw_rad),
        current_speed=speed_ms,
        waypoints=[(x1, y1), (x2, y2), ...],
        perception_data={'obstacles': [...], 'traffic_light': 'GREEN'},
    )
"""

from .pid_controller import LongitudinalPID, LateralPID
from .fsm_decision_maker import FSMDecisionMaker, VehicleState
from .vehicle_controller import VehicleController

__all__ = [
    "LongitudinalPID",
    "LateralPID",
    "FSMDecisionMaker",
    "VehicleState",
    "VehicleController",
]
