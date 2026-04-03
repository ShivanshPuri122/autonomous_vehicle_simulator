# perception/lane_detection/video_inference.py
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from ultralytics import YOLO

# Paths
ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'models' / 'lane_detection' / 'weights' / 'best.pt'
OUTPUT_DIR = ROOT / 'test_results' / 'video_output'

#  Confi
CONF_THRESHOLD = 0.30    
MAX_LANES      = 5       
PX_PER_M       = 108.0   

# Global variables for simple temporal smoothing (Lane Tracking)
prev_left_coeffs = None
prev_right_coeffs = None

LANE_COLORS = [
    (255, 100,  50), ( 50, 220, 100), (100,  50, 255),
    (255, 220,  50), (220,  50, 220)
]

#  Helper functions

def fit_polynomial(mask, degree=2):
    y_pts, x_pts = np.where(mask > 0)
    if len(y_pts) < 25:
        return None
    try:
        return np.polyfit(y_pts, x_pts, degree)
    except:
        return None

def poly_x(coeffs, y_val):
    if coeffs is None: return 0
    return np.polyval(coeffs, y_val)

def compute_curvature(coeffs, y_eval, px_per_m=PX_PER_M):
    if coeffs is None or len(coeffs) < 3: return 9999.0
    a_m, b_m = coeffs[0] / px_per_m, coeffs[1]
    y_m = y_eval / px_per_m
    d1, d2 = 2 * a_m * y_m + b_m, 2 * a_m
    if abs(d2) < 1e-9: return 9999.0 # Increased sensitivity
    return round(((1 + d1 ** 2) ** 1.5) / abs(d2), 1)

def compute_offset(left_c, right_c, img_w, img_h, px_per_m=PX_PER_M):
    y_b = img_h - 1
    if left_c is not None and right_c is not None:
        lane_cx = (poly_x(left_c, y_b) + poly_x(right_c, y_b)) / 2.0
    elif left_c is not None:
        lane_cx = poly_x(left_c, y_b) + img_w * 0.2
    elif right_c is not None:
        lane_cx = poly_x(right_c, y_b) - img_w * 0.2
    else: return None
    return round((img_w / 2.0 - lane_cx) / px_per_m, 3)

#Filters

def is_valid_mask_shape(mask, img_h, img_w):
    y_pts, x_pts = np.where(mask > 0)
    if len(y_pts) < 40: return False
    height = int(y_pts.max() - y_pts.min())
    width  = int(x_pts.max() - x_pts.min())
    # Reject short/wide shapes (intersection marks/cars)
    if height < img_h * 0.10: return False 
    if width > height * 1.6: return False
    return True

def is_valid_lane(coeffs, img_h, img_w):
    if coeffs is None: return False
    x_b = poly_x(coeffs, img_h - 1)
    x_t = poly_x(coeffs, img_h // 2)
    # Ensure line stays roughly on screen
    margin = img_w * 0.15
    if not (-margin <= x_b <= img_w + margin): return False
    # Reject horizontal-ish lines
    if abs(x_b - x_t) > img_h * 0.8: return False
    return True

# Drawing functions

def draw_drivable_area(image, left_c, right_c, img_h, img_w):
    global prev_left_coeffs, prev_right_coeffs
    
    # Simple Temporal Smoothing: If detection fails, use last known good lanes
    if left_c is None: left_c = prev_left_coeffs
    else: prev_left_coeffs = left_c
        
    if right_c is None: right_c = prev_right_coeffs
    else: prev_right_coeffs = right_c

    if left_c is None or right_c is None: return image
    
    # Safety: Ensure lanes aren't crossed or impossibly wide
    dist = abs(poly_x(right_c, img_h-1) - poly_x(left_c, img_h-1))
    if dist > img_w * 0.7 or dist < 20: return image

    overlay = image.copy()
    y_vals = np.linspace(img_h // 2, img_h - 1, 30).astype(int)
    l_pts = [(int(np.clip(poly_x(left_c, y), 0, img_w-1)), y) for y in y_vals]
    r_pts = [(int(np.clip(poly_x(right_c, y), 0, img_w-1)), y) for y in y_vals]
    pts = np.array(l_pts + r_pts[::-1], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (0, 200, 100))
    return cv2.addWeighted(image, 0.75, overlay, 0.25, 0)

# (draw_hud, draw_mask_overlay, draw_poly_curve remain largely the same)
def draw_mask_overlay(image, mask, color):
    colored = np.zeros_like(image); colored[mask > 0] = color
    return cv2.addWeighted(image, 1.0, colored, 0.35, 0)

def draw_poly_curve(image, coeffs, color, img_h, img_w, thickness=3):
    if coeffs is None: return image
    y_vals = np.arange(img_h // 2, img_h)
    x_vals = np.clip(np.polyval(coeffs, y_vals).astype(int), 0, img_w - 1)
    pts = np.column_stack([x_vals, y_vals]).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], False, color, thickness, cv2.LINE_AA)
    return image

def draw_hud(image, lane_stats, offset_m, fps):
    n_lanes = len(lane_stats); panel = image.copy()
    ph = 42 + n_lanes * 22 + 55
    cv2.rectangle(panel, (10, 10), (340, ph), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.55, image, 0.45, 0, image)
    font, y = cv2.FONT_HERSHEY_SIMPLEX, 32
    cv2.putText(image, f'FPS: {fps:.1f}', (18, y), font, 0.5, (80, 220, 80), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(image, f'Lanes detected: {n_lanes}', (18, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22
    for s in lane_stats:
        col = LANE_COLORS[s['id'] % len(LANE_COLORS)]
        r_str = f"{s['curvature']:.0f}m" if s['curvature'] < 9999 else "straight"
        cv2.putText(image, f"  L{s['id']}: curv={r_str}  conf={s['conf']:.2f}", (18, y), font, 0.42, col, 1, cv2.LINE_AA)
        y += 20
    if offset_m is not None:
        direction = 'LEFT' if offset_m < -0.1 else 'RIGHT' if offset_m > 0.1 else 'CENTER'
        cv2.putText(image, f'Offset: {offset_m:+.2f}m  [{direction}]', (18, y + 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image

#  Updated Per-frame processing

def process_frame(model, frame):
    h, w = frame.shape[:2]
    
    # ROI: Only look at the bottom 55% of the image to ignore cars/sky
    roi_y = int(h * 0.45) 
    frame_roi = frame[roi_y:, :]

    results = model.predict(source=frame_roi, conf=CONF_THRESHOLD, verbose=False)[0]

    lane_stats, all_coeffs = [], []

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for i, (mask, conf) in enumerate(zip(masks, confs)):
            # Resize mask back to ROI size then pad to full frame
            mask_r = cv2.resize(mask, (w, h - roi_y), interpolation=cv2.INTER_NEAREST).astype(np.uint8) * 255
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[roi_y:, :] = mask_r

            if not is_valid_mask_shape(full_mask, h, w): continue
            
            coeffs = fit_polynomial(full_mask, degree=2)
            if not is_valid_lane(coeffs, h, w): continue

            color = LANE_COLORS[len(lane_stats) % len(LANE_COLORS)]
            frame = draw_mask_overlay(frame, full_mask, color)
            
            all_coeffs.append(coeffs)
            lane_stats.append({
                'id': len(lane_stats), 'conf': float(conf),
                'curvature': compute_curvature(coeffs, h - 1), 'coeffs': coeffs
            })

    # Lane filtering and Drivable Area logic...
    left_c, right_c, cx = None, None, w / 2.0
    valid = sorted([(poly_x(c, h-1), c) for c in all_coeffs if c is not None], key=lambda t: t[0])
    
    left_cands = [c for x, c in valid if x < cx]
    right_cands = [c for x, c in valid if x >= cx]
    if left_cands: left_c = left_cands[-1]
    if right_cands: right_c = right_cands[0]

    frame = draw_drivable_area(frame, left_c, right_c, h, w)
    for i, s in enumerate(lane_stats):
        frame = draw_poly_curve(frame, s['coeffs'], LANE_COLORS[i % len(LANE_COLORS)], h, w)

    return frame, lane_stats, compute_offset(left_c, right_c, w, h)

# ── (run_video and main remain the same as previous) ──────────────────
def run_video(video_path, output_path):
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return
    fps_in, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (width, height))
    frame_count, t_start = 0, time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.time()
        annotated, stats, offset = process_frame(model, frame)
        fps = 1.0 / (time.time() - t0 + 1e-6)
        annotated = draw_hud(annotated, stats, offset, fps)
        writer.write(annotated)
        frame_count += 1
    cap.release(); writer.release()
    print(f"Done! Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    test_data_path = ROOT / 'data' / 'test_video'
    candidates = list(test_data_path.glob('*.mp4')) + list(ROOT.glob('*.mp4'))
    video_path = Path(args.video) if args.video else candidates[0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f'lane_{video_path.stem}_output.mp4'
    run_video(video_path, output_path)

if __name__ == '__main__':
    main()