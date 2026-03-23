import cv2
import mediapipe as mp
import numpy as np
from scipy.ndimage import gaussian_filter1d
import subprocess
import os


def convert_to_mp4(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        output_path
    ]

    subprocess.run(cmd, check=True)
    return output_path


def safe_crop_resize(frame, cx, cy, crop_w, crop_h, out_w, out_h):
    fh, fw = frame.shape[:2]

    target_ar = out_w / out_h
    crop_ar   = crop_w / crop_h if crop_h > 0 else target_ar

    if crop_ar > target_ar:
        crop_w = int(crop_h * target_ar)
    else:
        crop_h = int(crop_w / target_ar)

    crop_w = min(fw, crop_w)
    crop_h = min(fh, crop_h)

    x1 = int(np.clip(cx - crop_w / 2, 0, fw - crop_w))
    y1 = int(np.clip(cy - crop_h / 2, 0, fh - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = frame[y1:y2, x1:x2]

    if cropped.size == 0:
        cropped = frame

    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def process_video(input_path, output_path, final_path):
    print("🚀 Starting processing...")

    # Step 1: Convert input to clean MP4
    base, ext = os.path.splitext(input_path)
    clean_input = base + "_clean.mp4"
    INPUT_VIDEO = convert_to_mp4(input_path, clean_input)

    # CONFIG
    TARGET_RATIO     = 9 / 16
    SMOOTHING_SIGMA  = 2.5
    BASE_ZOOM_FACTOR = 1.2
    MIN_CROP_SCALE   = 0.70
    MAX_CROP_SCALE   = 0.95
    ZOOM_SMOOTH_MULT = 3.0
    PADDING_TOP      = 0.18

    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.45
    )

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {INPUT_VIDEO}")

    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    OUT_W, OUT_H = int(h * (9/16)), h
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input : {w}x{h} @ {fps:.2f} FPS | {total} frames")

    # ───────── PHASE 1: PRE-ANALYSIS ─────────
    raw_data   = []
    last_valid = [w / 2, h / 2, w * 0.15]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box

            xmin = max(0.0, bbox.xmin)
            ymin = max(0.0, bbox.ymin)
            bw   = min(bbox.width,  1.0 - xmin)
            bh   = min(bbox.height, 1.0 - ymin)

            cx = (xmin + bw / 2) * w
            cy = (ymin + bh / 2) * h - (bh * h * PADDING_TOP)
            fw = bw * w

            entry = [cx, cy, fw]
            last_valid = entry
        else:
            entry = last_valid.copy()

        raw_data.append(entry)

    print(f"Analysis complete: {len(raw_data)} frames")

    # ───────── PHASE 2: SMOOTHING ─────────
    raw_data  = np.array(raw_data)

    smooth_x  = gaussian_filter1d(raw_data[:, 0], sigma=SMOOTHING_SIGMA)
    smooth_y  = gaussian_filter1d(raw_data[:, 1], sigma=SMOOTHING_SIGMA)
    smooth_fw = gaussian_filter1d(raw_data[:, 2], sigma=SMOOTHING_SIGMA * ZOOM_SMOOTH_MULT)

    WARMUP_FRAMES = int(fps * 2)

    if len(smooth_x) > WARMUP_FRAMES:
        smooth_x[:WARMUP_FRAMES]  = smooth_x[WARMUP_FRAMES]
        smooth_y[:WARMUP_FRAMES]  = smooth_y[WARMUP_FRAMES]
        smooth_fw[:WARMUP_FRAMES] = smooth_fw[WARMUP_FRAMES]

    # ───────── PHASE 3: RENDERING ─────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (OUT_W, OUT_H))

    min_crop_h = int(h * MIN_CROP_SCALE)
    max_crop_h = int(h * MAX_CROP_SCALE)

    for i in range(len(smooth_x)):
        success, frame = cap.read()
        if not success:
            break

        raw_crop_h = int(smooth_fw[i] * BASE_ZOOM_FACTOR / TARGET_RATIO)
        raw_crop_h = int(raw_crop_h * 2)

        crop_h = int(np.clip(raw_crop_h, min_crop_h, max_crop_h))
        crop_w = int(crop_h * TARGET_RATIO)

        final = safe_crop_resize(
            frame,
            cx=smooth_x[i],
            cy=smooth_y[i],
            crop_w=crop_w,
            crop_h=crop_h,
            out_w=OUT_W,
            out_h=OUT_H
        )

        out.write(final)

    cap.release()
    out.release()

    # ───────── PHASE 4: MERGE AUDIO ─────────
    print("Merging audio...")

    cmd = f'''
    ffmpeg -y -i "{output_path}" -i "{INPUT_VIDEO}" \
    -c:v copy -c:a aac \
    -map 0:v:0 -map 1:a:0 \
    -shortest "{final_path}"
    '''

    os.system(cmd)

    print(f"✅ Done: {final_path}")

    return final_path