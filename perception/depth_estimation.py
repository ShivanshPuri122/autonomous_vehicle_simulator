"""
=============================================================================
Depth Estimation Module
=============================================================================

Fuses the CARLA depth camera output with object detection bounding boxes
to estimate the real-world distance (in metres) to each detected object.

The depth camera produces a per-pixel metric depth map that is aligned
with the RGB camera.  For each YOLO bounding box we sample the median
depth in the central region of the box (to avoid edge artefacts) and
report that as the object's distance.

Author : Perception Team
Version: 1.0.0
"""

import cv2
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  CORE API
# ═════════════════════════════════════════════════════════════════════════════

def estimate_obstacle_distances(depth_map, detections):
    """
    Estimate the distance to each detected object using the depth map.

    Parameters
    ----------
    depth_map : np.ndarray (H, W), float32
        Per-pixel depth in metres produced by the CARLA depth camera.
    detections : list[dict]
        Each dict must contain:
            'bbox'  : (x1, y1, x2, y2) — pixel coordinates of the bounding box
            'label' : str               — class name (e.g. 'vehicle', 'pedestrian')
            'confidence' : float        — detection confidence score
        Optional keys are passed through unchanged.

    Returns
    -------
    list[dict]
        Same dicts with two additional keys:
            'distance'       : float — median depth in metres (inf if unavailable)
            'lateral_offset' : float — signed horizontal offset from image centre (m)
    """
    if depth_map is None or len(detections) == 0:
        return detections

    img_h, img_w = depth_map.shape[:2]
    centre_x = img_w / 2.0

    enriched = []
    for det in detections:
        det = dict(det)  # avoid mutating the original

        x1, y1, x2, y2 = det['bbox']

        # Clamp to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img_w, int(x2))
        y2 = min(img_h, int(y2))

        if x2 <= x1 or y2 <= y1:
            det['distance'] = float('inf')
            det['lateral_offset'] = 0.0
            enriched.append(det)
            continue

        # Sample the central 50 % of the bounding box to avoid edge noise
        cx1 = x1 + (x2 - x1) // 4
        cx2 = x2 - (x2 - x1) // 4
        cy1 = y1 + (y2 - y1) // 4
        cy2 = y2 - (y2 - y1) // 4

        # Fall back to full box if the central region is too small
        if cx2 <= cx1 or cy2 <= cy1:
            cx1, cy1, cx2, cy2 = x1, y1, x2, y2

        roi = depth_map[cy1:cy2, cx1:cx2]

        if roi.size == 0:
            det['distance'] = float('inf')
        else:
            # Filter out sky / max-range pixels (CARLA returns ~1000 m for sky)
            valid = roi[roi < 999.0]
            if valid.size > 0:
                det['distance'] = round(float(np.median(valid)), 2)
            else:
                det['distance'] = float('inf')

        # Lateral offset: positive = object is to the RIGHT of centre
        bbox_centre_x = (x1 + x2) / 2.0
        det['lateral_offset'] = round(
            _pixel_to_lateral_metres(bbox_centre_x, centre_x, det['distance']),
            2
        )

        enriched.append(det)

    return enriched


def get_depth_at_point(depth_map, x, y):
    """
    Return the metric depth at a single pixel coordinate.

    Parameters
    ----------
    depth_map : np.ndarray (H, W), float32
    x, y      : int — pixel coordinates

    Returns
    -------
    float or None
    """
    if depth_map is None:
        return None
    h, w = depth_map.shape[:2]
    if 0 <= y < h and 0 <= x < w:
        return float(depth_map[y, x])
    return None


def create_depth_colormap(depth_map, max_range=100.0):
    """
    Create a colour-mapped visualisation of the depth map for debugging.

    Parameters
    ----------
    depth_map : np.ndarray (H, W), float32 — depth in metres
    max_range : float — maximum depth to visualise (metres)

    Returns
    -------
    np.ndarray (H, W, 3), uint8 — BGR colour image
    """
    if depth_map is None:
        return None

    # Normalise to [0, 255]
    normalised = np.clip(depth_map / max_range, 0.0, 1.0)
    grey = (255 * (1.0 - normalised)).astype(np.uint8)  # invert: close = bright

    # Apply a colourmap
    coloured = cv2.applyColorMap(grey, cv2.COLORMAP_MAGMA)
    return coloured


def draw_distance_annotations(image, detections, color=(0, 255, 255), thickness=2):
    """
    Draw bounding boxes and distance labels on an image.

    Parameters
    ----------
    image      : np.ndarray (H, W, 3) — BGR image to annotate (modified in-place)
    detections : list[dict]           — enriched detections with 'distance' key
    color      : tuple                — BGR colour for annotations
    thickness  : int                  — line thickness

    Returns
    -------
    np.ndarray — the annotated image
    """
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        dist = det.get('distance', float('inf'))
        label = det.get('label', '?')
        conf = det.get('confidence', 0.0)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Distance label
        if dist < 999.0:
            text = f"{label} {dist:.1f}m ({conf:.2f})"
        else:
            text = f"{label} --m ({conf:.2f})"

        # Background for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(
            image, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    return image


# ═════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _pixel_to_lateral_metres(pixel_x, centre_x, distance, fov_deg=90, img_width=640):
    """
    Convert a pixel x-offset from image centre to a lateral distance in metres,
    using pinhole camera geometry.

    Parameters
    ----------
    pixel_x   : float — horizontal pixel coordinate of the object centre
    centre_x  : float — horizontal pixel coordinate of the image centre
    distance  : float — depth to the object in metres
    fov_deg   : float — horizontal field of view in degrees
    img_width : int   — image width in pixels

    Returns
    -------
    float — signed lateral offset in metres (positive = right)
    """
    if distance <= 0 or distance >= 999.0:
        return 0.0

    # Focal length in pixels from FOV
    fov_rad = np.deg2rad(fov_deg)
    focal_px = (img_width / 2.0) / np.tan(fov_rad / 2.0)

    # Angular offset
    dx_px = pixel_x - centre_x
    lateral_m = (dx_px / focal_px) * distance

    return lateral_m
