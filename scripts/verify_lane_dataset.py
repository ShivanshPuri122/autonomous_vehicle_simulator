# scripts/verify_dataset.py
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
splits = ['train', 'val', 'test']

print('DATASET VERIFICATION')
print('─' * 55)

all_good = True

for split in splits:
    label_dir = ROOT / 'data' / 'lane_data' / split / 'labels'
    img_dir   = ROOT / 'data' / 'lane_data' / split / 'images'

    txts = list(label_dir.glob('*.txt'))
    imgs = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

    counts = []
    empty  = 0
    for f in txts:
        lines = [l for l in f.read_text().strip().splitlines() if l]
        counts.append(len(lines))
        if len(lines) == 0:
            empty += 1

    if counts:
        avg = sum(counts) / len(counts)
        mn  = min(counts)
        mx  = max(counts)
        ok  = 'Ok' if avg >= 2.0 else 'X'
        if avg < 2.0:
            all_good = False
        print(f'{split:6} | images: {len(imgs):5} | labels: {len(txts):5} | '
              f'avg lanes: {avg:.2f} {ok} | min: {mn} max: {mx} | empty: {empty}')
    else:
        all_good = False
        print(f'{split:6} | No label files found in {label_dir}')

print('─' * 55)
if all_good:
    print('Dataset verified — ready to zip and upload to Colab!')
else:
    print(' Issues found — fix before training')