"""
Standalone test for the depth estimation module.
Tests the core logic with synthetic depth maps — no CARLA required.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perception.depth_estimation import (
    estimate_obstacle_distances,
    get_depth_at_point,
    create_depth_colormap,
    _pixel_to_lateral_metres,
)


def test_depth_at_point():
    """Test single-pixel depth query."""
    depth_map = np.full((480, 640), 25.0, dtype=np.float32)
    depth_map[100, 200] = 10.5

    assert get_depth_at_point(depth_map, 200, 100) == 10.5, "Pixel depth mismatch"
    assert get_depth_at_point(depth_map, 0, 0) == 25.0, "Default depth mismatch"
    assert get_depth_at_point(None, 0, 0) is None, "None map should return None"
    assert get_depth_at_point(depth_map, -1, 0) is None, "OOB should return None"
    assert get_depth_at_point(depth_map, 0, 999) is None, "OOB should return None"
    print("[PASS] test_depth_at_point")


def test_estimate_distances_basic():
    """Test distance estimation with a uniform depth map."""
    depth_map = np.full((480, 640), 15.0, dtype=np.float32)

    detections = [
        {'bbox': (100, 100, 200, 200), 'label': 'vehicle', 'confidence': 0.9},
        {'bbox': (400, 300, 500, 400), 'label': 'pedestrian', 'confidence': 0.85},
    ]

    enriched = estimate_obstacle_distances(depth_map, detections)

    assert len(enriched) == 2
    assert enriched[0]['distance'] == 15.0, f"Expected 15.0, got {enriched[0]['distance']}"
    assert enriched[1]['distance'] == 15.0, f"Expected 15.0, got {enriched[1]['distance']}"
    assert 'lateral_offset' in enriched[0]
    assert 'lateral_offset' in enriched[1]
    print("[PASS] test_estimate_distances_basic")


def test_estimate_distances_variable_depth():
    """Test that median depth is correctly computed for a region with varying depths."""
    depth_map = np.full((480, 640), 50.0, dtype=np.float32)
    # Place an object at 10m in a region
    depth_map[150:250, 150:250] = 10.0

    detections = [
        {'bbox': (150, 150, 250, 250), 'label': 'vehicle', 'confidence': 0.9},
    ]

    enriched = estimate_obstacle_distances(depth_map, detections)
    assert enriched[0]['distance'] == 10.0, f"Expected 10.0, got {enriched[0]['distance']}"
    print("[PASS] test_estimate_distances_variable_depth")


def test_sky_filtering():
    """Test that sky pixels (depth >= 999m) are filtered out."""
    depth_map = np.full((480, 640), 999.5, dtype=np.float32)
    # Place real depth in the central region
    depth_map[160:190, 160:190] = 20.0

    detections = [
        {'bbox': (150, 150, 200, 200), 'label': 'vehicle', 'confidence': 0.8},
    ]

    enriched = estimate_obstacle_distances(depth_map, detections)
    # The central 50% of the bbox should include some 20.0m pixels
    assert enriched[0]['distance'] < 999.0, f"Sky should be filtered, got {enriched[0]['distance']}"
    print("[PASS] test_sky_filtering")


def test_empty_inputs():
    """Test graceful handling of empty inputs."""
    depth_map = np.full((480, 640), 25.0, dtype=np.float32)

    # Empty detections
    result = estimate_obstacle_distances(depth_map, [])
    assert result == [], "Empty detections should return empty list"

    # None depth map
    detections = [{'bbox': (100, 100, 200, 200), 'label': 'test', 'confidence': 0.5}]
    result = estimate_obstacle_distances(None, detections)
    assert result == detections, "None depth map should return original detections"

    print("[PASS] test_empty_inputs")


def test_lateral_offset():
    """Test lateral offset computation."""
    # Object at image centre → offset should be ~0
    offset = _pixel_to_lateral_metres(320.0, 320.0, 10.0, fov_deg=90, img_width=640)
    assert abs(offset) < 0.01, f"Centre offset should be ~0, got {offset}"

    # Object to the right
    offset = _pixel_to_lateral_metres(480.0, 320.0, 10.0, fov_deg=90, img_width=640)
    assert offset > 0, f"Right-of-centre should be positive, got {offset}"

    # Object to the left
    offset = _pixel_to_lateral_metres(160.0, 320.0, 10.0, fov_deg=90, img_width=640)
    assert offset < 0, f"Left-of-centre should be negative, got {offset}"

    print("[PASS] test_lateral_offset")


def test_depth_colormap():
    """Test that colormap generation doesn't crash."""
    depth_map = np.random.uniform(0, 100, (480, 640)).astype(np.float32)
    coloured = create_depth_colormap(depth_map, max_range=100.0)

    assert coloured is not None
    assert coloured.shape == (480, 640, 3)
    assert coloured.dtype == np.uint8

    # None input
    assert create_depth_colormap(None) is None

    print("[PASS] test_depth_colormap")


if __name__ == "__main__":
    print("=" * 60)
    print("Depth Estimation Module — Unit Tests")
    print("=" * 60)

    test_depth_at_point()
    test_estimate_distances_basic()
    test_estimate_distances_variable_depth()
    test_sky_filtering()
    test_empty_inputs()
    test_lateral_offset()
    test_depth_colormap()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
