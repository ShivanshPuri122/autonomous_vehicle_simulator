import os
import cv2
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

MODEL_PATH  = r"models\object_detection\weights\best.pt"
VIDEO_PATH  = r"your_carla_video.mp4"      
DATA_YAML   = r"config\object_data.yaml"
OUTPUT_DIR  = r"presentation_outputs"

CONF_THRESH = 0.25
IMGSZ       = 512

# ─────────────────────────────────────────────────────────────
#  CLASS DEFINITIONS
# ─────────────────────────────────────────────────────────────

CLASSES = [
    'vehicle', 'bike',
    'traffic_light_red', 'traffic_light_green',
    'traffic_light_yellow', 'traffic_light_off',
    'speed_sign_30', 'speed_sign_60', 'speed_sign_90',
    'traffic_sign', 'pedestrian'
]

COLORS_BGR = {
    'vehicle'              : (0,   165, 255),
    'bike'                 : (0,   255, 255),
    'traffic_light_red'    : (0,   0,   255),
    'traffic_light_green'  : (0,   255, 0  ),
    'traffic_light_yellow' : (0,   255, 255),
    'traffic_light_off'    : (128, 128, 128),
    'speed_sign_30'        : (255, 0,   255),
    'speed_sign_60'        : (200, 0,   255),
    'speed_sign_90'        : (255, 0,   200),
    'traffic_sign'         : (255, 165, 0  ),
    'pedestrian'           : (255, 255, 0  ),
}

COLORS_RGB = {
    k: (v[2]/255, v[1]/255, v[0]/255)
    for k, v in COLORS_BGR.items()
}

AV_THRESHOLDS = {
    'vehicle'              : 0.95,
    'bike'                 : 0.93,
    'traffic_light_red'    : 0.95,
    'traffic_light_green'  : 0.95,
    'traffic_light_yellow' : 0.95,
    'traffic_light_off'    : 0.90,
    'speed_sign_30'        : 0.93,
    'speed_sign_60'        : 0.93,
    'speed_sign_90'        : 0.93,
    'traffic_sign'         : 0.90,
    'pedestrian'           : 0.95,
}

# ─────────────────────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────────────────────

def setup():
    """Create output dirs and check GPU."""
    for d in ['report', 'plots', 'video']:
        Path(f'{OUTPUT_DIR}/{d}').mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 0:
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'✅ GPU: {gpu} ({vram:.1f}GB VRAM)')
    else:
        print('⚠️  No GPU — using CPU (slower)')

    print(f'📁 Outputs → {OUTPUT_DIR}/')
    return device

# ─────────────────────────────────────────────────────────────
#  STEP 1 — METRICS REPORT
# ─────────────────────────────────────────────────────────────

def generate_metrics_report(model, device):
    """Run validation and generate full metrics report."""
    print('\n' + '='*60)
    print('  📊 STEP 1 — Generating Metrics Report')
    print('='*60)

    metrics = model.val(
        data    = DATA_YAML,
        split   = 'test',
        device  = device,
        imgsz   = IMGSZ,
        batch   = 8,
        plots   = True,
        verbose = True,
    )

    # ── Build report ─────────────────────────────────────────
    report = {
        'model_info': {
            'architecture' : 'YOLOv8m',
            'dataset'      : 'CARLA Merged (28,814 images)',
            'classes'      : 11,
            'generated_at' : datetime.now().strftime('%Y-%m-%d %H:%M'),
        },
        'overall': {
            'mAP50'     : round(float(metrics.box.map50), 4),
            'mAP50_95'  : round(float(metrics.box.map),   4),
            'precision' : round(float(metrics.box.mp),    4),
            'recall'    : round(float(metrics.box.mr),    4),
        },
        'per_class'    : {},
        'av_readiness' : {},
    }

    # ✅ Fixed — safely handle missing classes in ap50
    ap50_list = list(metrics.box.ap50)
    passed    = []
    failed    = []

    for i, name in enumerate(CLASSES):
        # Safely get ap value
        ap_val    = round(float(ap50_list[i]), 4) \
                    if i < len(ap50_list) else 0.0
        threshold = AV_THRESHOLDS.get(name, 0.90)
        status    = 'PASS' if ap_val >= threshold else 'FAIL'

        report['per_class'][name]    = ap_val
        report['av_readiness'][name] = {
            'mAP50'   : ap_val,
            'required': threshold,
            'status'  : status,
        }

        if status == 'PASS':
            passed.append(name)
        else:
            failed.append(name)

    # ── Print results ─────────────────────────────────────────
    print('\n' + '='*60)
    print('  📈 OVERALL RESULTS')
    print('='*60)
    print(f'  mAP50      : {report["overall"]["mAP50"]:.4f}')
    print(f'  mAP50-95   : {report["overall"]["mAP50_95"]:.4f}')
    print(f'  Precision  : {report["overall"]["precision"]:.4f}')
    print(f'  Recall     : {report["overall"]["recall"]:.4f}')

    print('\n' + '='*60)
    print('  🚗 PER CLASS AV READINESS')
    print('='*60)
    print(f'  {"Class":<25} {"mAP50":<8} {"Required":<10} {"Status"}')
    print('  ' + '-'*55)

    for name in CLASSES:
        data   = report['av_readiness'][name]
        status = '✅ PASS' if data['status'] == 'PASS' else '❌ FAIL'
        print(f'  {name:<25} {data["mAP50"]:<8.4f} '
              f'{data["required"]:<10.2f} {status}')

    print(f'\n  Passed : {len(passed)}/{len(CLASSES)} classes')
    if failed:
        print(f'  Failed : {failed}')

    # ── Save JSON ─────────────────────────────────────────────
    report_path = f'{OUTPUT_DIR}/report/metrics_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n✅ Report saved → {report_path}')

    # ── Copy validation plots ─────────────────────────────────
    import shutil
    # Find the latest val folder
    val_dirs = sorted(Path('runs/detect').glob('val*'),
                      key=os.path.getmtime, reverse=True)
    if val_dirs:
        val_dir = val_dirs[0]
        for plot in ['confusion_matrix.png', 'confusion_matrix_normalized.png',
                     'PR_curve.png', 'F1_curve.png',
                     'P_curve.png', 'R_curve.png', 'results.png']:
            src = val_dir / plot
            if src.exists():
                dst = f'{OUTPUT_DIR}/plots/{plot}'
                shutil.copy(src, dst)
                print(f'✅ Copied {plot}')

    return metrics, report

# ─────────────────────────────────────────────────────────────
#  STEP 2 — GENERATE CHARTS
# ─────────────────────────────────────────────────────────────

def generate_charts(report):
    """Generate presentation-quality dark theme charts."""
    print('\n' + '='*60)
    print('  📊 STEP 2 — Generating Charts')
    print('='*60)

    # ── Chart 1 — Per Class mAP50 Bar Chart ──────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    names   = CLASSES
    values  = [report['per_class'].get(n, 0) for n in names]
    colors  = [COLORS_RGB.get(n, (0.5, 0.5, 0.5)) for n in names]
    bars    = ax.barh(names, values, color=colors,
                      edgecolor='white', linewidth=0.5, height=0.6)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.001,
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f}',
                va='center', ha='left',
                color='white', fontsize=9, fontweight='bold')

    # Threshold lines
    ax.axvline(x=0.95, color='#ff4444', linestyle='--',
               linewidth=2, label='AV Safety Threshold (0.95)')
    ax.axvline(x=0.90, color='#ffaa00', linestyle='--',
               linewidth=2, label='Minimum Threshold (0.90)')

    ax.set_xlabel('mAP50', color='white', fontsize=12)
    ax.set_title('Per-Class Object Detection Performance (mAP50)\n'
                 'CARLA Autonomous Vehicle Dataset',
                 color='white', fontsize=13,
                 fontweight='bold', pad=15)
    ax.tick_params(colors='white', labelsize=10)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.85, 1.03)
    ax.legend(facecolor='#1a1a2e', labelcolor='white',
              fontsize=9, loc='lower right',
              framealpha=0.8)

    # Add grid
    ax.xaxis.grid(True, color='#333', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path1 = f'{OUTPUT_DIR}/plots/per_class_mAP50.png'
    plt.savefig(path1, dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f'✅ Chart 1 → {path1}')

    # ── Chart 2 — Overall Metrics + AV Readiness ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')

    # Left — Overall metrics bars
    ax1 = axes[0]
    ax1.set_facecolor('#16213e')

    m_names  = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
    m_values = [
        report['overall']['mAP50'],
        report['overall']['mAP50_95'],
        report['overall']['precision'],
        report['overall']['recall'],
    ]
    m_colors = ['#00ff88', '#00ccff', '#ff9900', '#ff4466']
    bars2    = ax1.bar(m_names, m_values,
                       color=m_colors, edgecolor='white',
                       linewidth=0.5, width=0.5)

    for bar, val in zip(bars2, m_values):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f'{val:.4f}',
                 ha='center', va='bottom',
                 color='white', fontsize=12,
                 fontweight='bold')

    ax1.set_ylim(0.60, 1.05)
    ax1.set_title('Overall Model Performance',
                  color='white', fontsize=13,
                  fontweight='bold', pad=10)
    ax1.tick_params(colors='white', labelsize=10)
    ax1.spines['bottom'].set_color('#444')
    ax1.spines['left'].set_color('#444')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axhline(y=0.95, color='red', linestyle='--',
                alpha=0.7, linewidth=1.5,
                label='Target (0.95)')
    ax1.yaxis.grid(True, color='#333', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(facecolor='#1a1a2e', labelcolor='white',
               fontsize=9)

    # Right — AV Readiness pie
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')

    passed = sum(1 for v in report['av_readiness'].values()
                 if v['status'] == 'PASS')
    failed = len(CLASSES) - passed

    wedges, texts, autotexts = ax2.pie(
        [passed, failed],
        labels=[f'Ready ✅\n({passed} classes)',
                f'Needs Work ⚠️\n({failed} classes)'],
        colors=['#00ff88', '#ff4466'],
        autopct='%1.0f%%',
        startangle=90,
        textprops={'color': 'white', 'fontsize': 11},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
        at.set_fontsize(12)

    ax2.set_title('AV Production Readiness\n(Per-Class Safety Check)',
                  color='white', fontsize=13,
                  fontweight='bold', pad=10)

    plt.tight_layout()
    path2 = f'{OUTPUT_DIR}/plots/overall_metrics.png'
    plt.savefig(path2, dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f'✅ Chart 2 → {path2}')

    # ── Chart 3 — Class Recall Comparison ────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # Known recall values from your validation
    recall_values = {
        'vehicle'              : 0.914,
        'bike'                 : 0.893,
        'traffic_light_red'    : 0.859,
        'traffic_light_green'  : 0.909,
        'traffic_light_yellow' : 0.907,
        'traffic_light_off'    : 0.900,
        'speed_sign_30'        : 0.969,
        'speed_sign_60'        : 0.922,
        'speed_sign_90'        : 0.918,
        'traffic_sign'         : 0.914,
        'pedestrian'           : 0.931,
    }

    r_names  = list(recall_values.keys())
    r_values = list(recall_values.values())
    r_colors = ['#ff4444' if v < 0.90
                else '#ffaa00' if v < 0.95
                else '#00ff88'
                for v in r_values]

    bars3 = ax.bar(range(len(r_names)), r_values,
                   color=r_colors, edgecolor='white',
                   linewidth=0.5, width=0.6)

    for bar, val in zip(bars3, r_values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{val:.3f}',
                ha='center', va='bottom',
                color='white', fontsize=8,
                fontweight='bold')

    ax.set_xticks(range(len(r_names)))
    ax.set_xticklabels(r_names, rotation=35,
                       ha='right', color='white', fontsize=9)
    ax.set_ylabel('Recall', color='white', fontsize=12)
    ax.set_title('Per-Class Recall\n'
                 '(Critical for AV Safety)',
                 color='white', fontsize=13,
                 fontweight='bold', pad=10)
    ax.set_ylim(0.80, 1.02)
    ax.axhline(y=0.95, color='#ff4444', linestyle='--',
               linewidth=2, label='AV Safety Target (0.95)')
    ax.axhline(y=0.90, color='#ffaa00', linestyle='--',
               linewidth=2, label='Minimum (0.90)')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#333', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(facecolor='#1a1a2e', labelcolor='white',
              fontsize=9)

    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00ff88', label='Above target (>0.95)'),
        Patch(facecolor='#ffaa00', label='Acceptable (0.90-0.95)'),
        Patch(facecolor='#ff4444', label='Below minimum (<0.90)'),
    ]
    ax.legend(handles=legend_elements,
              facecolor='#1a1a2e', labelcolor='white',
              fontsize=9, loc='lower right')

    plt.tight_layout()
    path3 = f'{OUTPUT_DIR}/plots/per_class_recall.png'
    plt.savefig(path3, dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f'✅ Chart 3 → {path3}')

# ─────────────────────────────────────────────────────────────
#  STEP 3 — ANNOTATED DEMO VIDEO
# ─────────────────────────────────────────────────────────────

def generate_demo_video(model, device):
    """Run model on CARLA video and save annotated output."""
    print('\n' + '='*60)
    print('  🎥 STEP 3 — Generating Demo Video')
    print('='*60)

    if not Path(VIDEO_PATH).exists():
        print(f'❌ Video not found: {VIDEO_PATH}')
        print('   Update VIDEO_PATH in config section')
        print('   Skipping video generation...')
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f'❌ Cannot open video: {VIDEO_PATH}')
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'   Video  : {total_frames} frames | '
          f'{fps:.1f}fps | {width}x{height}')

    output_path = f'{OUTPUT_DIR}/video/demo_annotated.mp4'
    fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
    out         = cv2.VideoWriter(
        output_path, fourcc, fps, (width, height))

    frame_count  = 0
    detect_count = 0

    print('   Processing...')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ── Run inference ─────────────────────────────────────
        results = model.predict(
            source  = frame,
            conf    = CONF_THRESH,
            imgsz   = IMGSZ,
            device  = device,
            verbose = False,
        )

        annotated           = frame.copy()
        traffic_light_state = None
        speed_limit         = None
        frame_dets          = 0

        for result in results:
            for box in result.boxes:
                cls_id       = int(box.cls[0])
                name         = CLASSES[cls_id] \
                               if cls_id < len(CLASSES) \
                               else f'class_{cls_id}'
                conf         = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                color        = COLORS_BGR.get(
                    name, (255, 255, 255))

                # Draw box
                cv2.rectangle(annotated,
                    (x1, y1), (x2, y2), color, 2)

                # Label background
                label       = f'{name} {conf:.2f}'
                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated,
                    (x1, y1-th-6), (x1+tw+4, y1),
                    color, -1)
                cv2.putText(annotated, label,
                    (x1+2, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1,
                    cv2.LINE_AA)

                # Track traffic light
                if name == 'traffic_light_red':
                    traffic_light_state = ('RED',    (0,   0,   255))
                elif name == 'traffic_light_green':
                    traffic_light_state = ('GREEN',  (0,   255, 0  ))
                elif name == 'traffic_light_yellow':
                    traffic_light_state = ('YELLOW', (0,   255, 255))

                # Track speed limit
                speed_map = {
                    'speed_sign_30': 30,
                    'speed_sign_60': 60,
                    'speed_sign_90': 90,
                }
                if name in speed_map:
                    speed_limit = speed_map[name]

                frame_dets   += 1
                detect_count += 1

        # ── HUD Top Bar ───────────────────────────────────────
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, 75),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, annotated,
                        0.35, 0, annotated)

        cv2.putText(annotated,
            'YOLOv8m | CARLA AV Object Detection',
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(annotated,
            f'mAP50: 0.965 | Classes: 11 | '
            f'Detections: {frame_dets}',
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(annotated,
            f'{frame_count}/{total_frames}',
            (width-130, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (180, 180, 180), 1, cv2.LINE_AA)

        # ── HUD Bottom Bar ────────────────────────────────────
        overlay2 = annotated.copy()
        cv2.rectangle(overlay2,
            (0, height-65), (width, height),
            (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.65, annotated,
                        0.35, 0, annotated)

        # Traffic light state
        if traffic_light_state:
            state, color = traffic_light_state
            cv2.putText(annotated,
                f'Traffic Light: {state}',
                (10, height-35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated,
                'Traffic Light: NONE',
                (10, height-35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (128, 128, 128), 1, cv2.LINE_AA)

        # Speed limit
        if speed_limit:
            cv2.putText(annotated,
                f'Speed Limit: {speed_limit} km/h',
                (10, height-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 165, 255), 2, cv2.LINE_AA)

        out.write(annotated)

        # Progress
        if frame_count % 50 == 0:
            pct = frame_count / total_frames * 100
            print(f'   {pct:5.1f}% — '
                  f'{frame_count}/{total_frames} frames')

    cap.release()
    out.release()

    print(f'\n✅ Video saved → {output_path}')
    print(f'   Frames    : {frame_count}')
    print(f'   Detections: {detect_count}')
    print(f'   Avg/frame : {detect_count/max(frame_count,1):.1f}')

# ─────────────────────────────────────────────────────────────
#  STEP 4 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

def print_summary(report):
    passed = sum(1 for v in report['av_readiness'].values()
                 if v['status'] == 'PASS')
    failed = len(CLASSES) - passed

    print('\n' + '='*60)
    print('  🎉 PRESENTATION READY!')
    print('='*60)
    print(f'\n  📁 {OUTPUT_DIR}/')
    print(f'  ├── report/')
    print(f'  │   └── metrics_report.json')
    print(f'  ├── plots/')
    print(f'  │   ├── per_class_mAP50.png      ← slides ✅')
    print(f'  │   ├── per_class_recall.png     ← slides ✅')
    print(f'  │   ├── overall_metrics.png      ← slides ✅')
    print(f'  │   ├── confusion_matrix.png     ← slides ✅')
    print(f'  │   └── PR_curve.png             ← slides ✅')
    print(f'  └── video/')
    print(f'      └── demo_annotated.mp4       ← panel demo ✅')

    print(f'\n  📊 Key Numbers:')
    print(f'     mAP50      : {report["overall"]["mAP50"]:.4f}')
    print(f'     mAP50-95   : {report["overall"]["mAP50_95"]:.4f}')
    print(f'     Precision  : {report["overall"]["precision"]:.4f}')
    print(f'     Recall     : {report["overall"]["recall"]:.4f}')
    print(f'     Classes OK : {passed}/{len(CLASSES)}')

    print(f'\n  🚗 AV Readiness:')
    for name, data in report['av_readiness'].items():
        icon = '✅' if data['status'] == 'PASS' else '❌'
        print(f'     {icon} {name:<25} {data["mAP50"]:.4f}')

    print('='*60)

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('='*60)
    print('  🚗 AV Object Detection — Presentation Pipeline')
    print('='*60)

    # Setup
    device = setup()

    # Load model
    print(f'\n📦 Loading: {MODEL_PATH}')
    if not Path(MODEL_PATH).exists():
        print(f'❌ Model not found!')
        print(f'   Place best.pt in: {MODEL_PATH}')
        exit()

    model = YOLO(MODEL_PATH)
    print('✅ Model loaded!')

    # Step 1 — Metrics
    metrics, report = generate_metrics_report(model, device)

    # Step 2 — Charts
    generate_charts(report)

    # Step 3 — Video
    generate_demo_video(model, device)

    # Step 4 — Summary
    print_summary(report)
