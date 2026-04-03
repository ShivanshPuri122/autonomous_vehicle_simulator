# scripts/tusimple_to_yolo.py
#
# What this does:
#   Reads paired (image, mask) from TuSimple preprocessed dataset
#   Converts each binary mask → YOLO segmentation polygon labels
#   Splits into train/val/test (80/10/10)
#   Writes everything into data/lane_data/ ready for training

import cv2
import numpy as np
import shutil
import random
from pathlib import Path

# Paths 
ROOT = Path(__file__).resolve().parent.parent 

TUSIMPLE_FRAMES = ROOT / 'data' / 'raw' / 'tusimple' / 'tusimple_preprocessed' / 'training' / 'frames'
TUSIMPLE_MASKS  = ROOT / 'data' / 'raw' / 'tusimple' / 'tusimple_preprocessed' / 'training' / 'lane-masks'

OUT_DIR = ROOT / 'data' / 'lane_data'


# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# Minimum contour area in pixels — filters out noise dots

MIN_AREA = 80


def mask_to_yolo_polygons(mask_path):
    """
    Converts a binary lane mask PNG to YOLO segmentation format.

    The mask has:
      - Black pixels (0)   = background
      - White pixels (255) = lane marking

    Each separate white region = one lane instance.

    Steps:
      1. Read mask as grayscale
      2. Threshold to strict binary (handles any near-white values)
      3. Erode slightly to separate touching lanes
      4. Find individual contours — one per lane
      5. Simplify each contour to fewer points
      6. Normalize coordinates to [0, 1]
      7. Format as YOLO seg string: "0 x1 y1 x2 y2 ..."
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], 0

    h, w = mask.shape

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if binary.sum() == 0:
        return [], 0

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    yolo_lines = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue

        # Sample ~20 points evenly along contour
        step = max(1, len(cnt) // 20)
        pts  = cnt[::step].reshape(-1, 2).astype(np.float32)

        if len(pts) < 3:
            continue

        pts[:, 0] /= w
        pts[:, 1] /= h
        pts = np.clip(pts, 0.0, 1.0)

        coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in pts)
        yolo_lines.append(f'0 {coords}')

    return yolo_lines, len(yolo_lines)


def convert():
    # Clean previous output first 
    import shutil as sh
    if OUT_DIR.exists():
        sh.rmtree(OUT_DIR)
        print('Cleaned previous conversion.')
    # Get all image files
    all_frames = sorted(TUSIMPLE_FRAMES.glob('*.jpg'))

    if not all_frames:
        print(f'No images found in {TUSIMPLE_FRAMES}')
        return

    print(f'Found {len(all_frames)} images')

    # Verify masks exist
    # Check first 5 to confirm naming matches
    print('\nVerifying mask pairing...')
    for f in all_frames[:5]:
        mask_path = TUSIMPLE_MASKS / f.name
        print(f'  {f.name} → mask exists: {mask_path.exists()}')

    #Shuffle and split
    random.seed(42)  
    frames = all_frames.copy()
    random.shuffle(frames)

    total      = len(frames)
    val_end    = int(total * VAL_RATIO)
    test_end   = int(total * (VAL_RATIO + TEST_RATIO))

    splits = {
        'val'  : frames[:val_end],       
        'test' : frames[val_end:test_end],    
        'train': frames[test_end:],         
    }

    print(f'\nSplit sizes:')
    for s, files in splits.items():
        print(f'  {s:6}: {len(files)} images')

    # ── Create output directories ─────────────────────────────
    for split in ['train', 'val', 'test']:
        (OUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    #Convert and copy
    print('\nConverting...')

    inst_counts  = []   # track instances per image for summary
    missing_masks = 0
    total_done   = 0

    for split, file_list in splits.items():
        for img_path in file_list:
            mask_path = TUSIMPLE_MASKS / img_path.name

            if not mask_path.exists():
                missing_masks += 1
                continue

            # Convert mask to YOLO polygons
            yolo_lines, n_inst = mask_to_yolo_polygons(mask_path)

            # Destination paths
            dest_img = OUT_DIR / split / 'images' / img_path.name
            dest_lbl = OUT_DIR / split / 'labels' / \
                       img_path.with_suffix('.txt').name

            # Copy image
            shutil.copy2(str(img_path), str(dest_img))

            dest_lbl.write_text('\n'.join(yolo_lines), encoding='utf-8')

            inst_counts.append(n_inst)
            total_done += 1

            if total_done % 300 == 0:
                print(f'  [{total_done}/{total}] processed...')


    print(f'\n{"─" * 50}')
    print('CONVERSION COMPLETE')
    print(f'{"─" * 50}')
    print(f'Processed     : {total_done}')
    print(f'Missing masks : {missing_masks}')

    if inst_counts:
        avg = sum(inst_counts) / len(inst_counts)
        zeros = sum(1 for c in inst_counts if c == 0)
        ones  = sum(1 for c in inst_counts if c == 1)
        multi = sum(1 for c in inst_counts if c >= 2)

        print(f'\nAvg lanes/image : {avg:.2f}  (target: 2-4)')
        print(f'0 instances     : {zeros}')
        print(f'1 instance      : {ones}')
        print(f'2+ instances    : {multi} ')

        if avg >= 2.0:
            print('\nDataset looks good ready to train!')
        else:
            print('\nLow instance count check mask quality')


if __name__ == '__main__':
    convert()