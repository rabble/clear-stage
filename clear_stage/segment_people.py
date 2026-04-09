"""Segment multiple people using SAM2 with per-person object tracking."""
import os
import cv2
import numpy as np
import torch


def segment_background_people(
    video_path: str,
    detections: list[dict],
    principal_idx: int,
    sam2_checkpoint: str,
    frame_idx: int = 0,
    device: str = "cuda",
    erode_iterations: int = 2,
) -> np.ndarray:
    """
    Segment all background people (everyone except the principal) using SAM2.

    Each person gets their own SAM2 object ID and bounding box prompt,
    then all are tracked through the video simultaneously.

    Returns: binary mask video as numpy array (T, H, W) uint8
             where 0 = remove (background person), 255 = keep
    """
    from sam2.build_sam import build_sam2_video_predictor
    import tempfile

    # Extract frames to a temp directory (SAM2 needs frame images)
    frames_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{frame_count:06d}.jpg"), frame)
        frame_count += 1
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    print(f"Extracted {frame_count} frames ({w}x{h})")

    # Build SAM2 predictor
    predictor = build_sam2_video_predictor(
        "sam2_hiera_l.yaml", sam2_checkpoint, device=device
    )

    # Initialize video state
    state = predictor.init_state(video_path=frames_dir)

    # Add each background person as a separate object with their bounding box
    bg_people = [(i, d) for i, d in enumerate(detections) if i != principal_idx]
    print(f"Tracking {len(bg_people)} background people as separate objects")

    for obj_id, (det_idx, det) in enumerate(bg_people):
        x1, y1, x2, y2 = det["bbox"]
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        # Add bounding box prompt for this person on the detection frame
        _, _, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )
        print(f"  Object {obj_id} (person #{det_idx}): box={det['bbox']}")

    # Propagate through entire video
    print("Propagating masks through video...")
    all_masks = {}  # frame_idx -> combined mask

    for frame_idx_out, obj_ids, masks in predictor.propagate_in_video(state):
        # Combine all object masks for this frame
        combined = np.zeros((h, w), dtype=bool)
        for obj_id_out, mask in zip(obj_ids, masks):
            mask_np = (mask[0] > 0.0).cpu().numpy()
            # Resize if needed
            if mask_np.shape != (h, w):
                mask_np = cv2.resize(mask_np.astype(np.uint8), (w, h),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
            combined = combined | mask_np
        all_masks[frame_idx_out] = combined

    # Build output mask video: 0 where people are (remove), 255 elsewhere (keep)
    mask_video = np.full((frame_count, h, w), 255, dtype=np.uint8)
    for fidx, mask in all_masks.items():
        mask_video[fidx][mask] = 0

    # Cleanup
    predictor.reset_state(state)
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)

    # Dilate keep regions (255) to shrink removal regions (0),
    # preventing the mask from eating into the principal dancer.
    # Skip erosion if total removal area is small (< 5% of frame),
    # as erosion can completely erase small masks.
    if erode_iterations > 0:
        removal_pct = np.sum(mask_video[0] == 0) / mask_video[0].size
        if removal_pct > 0.05:  # Only erode if significant removal area
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            for i in range(mask_video.shape[0]):
                mask_video[i] = cv2.dilate(mask_video[i], kernel, iterations=erode_iterations)
            print(f"Eroded masks ({erode_iterations} iterations)")
        else:
            print(f"Skipped erosion (removal area {removal_pct*100:.1f}% too small)")

    print(f"Mask video: {mask_video.shape}, "
          f"removal coverage frame 0: {np.sum(mask_video[0] == 0) / mask_video[0].size * 100:.1f}%")
    return mask_video


def save_mask_video(mask_video: np.ndarray, output_path: str, fps: float = 30.0) -> str:
    """Save mask numpy array as video file."""
    import subprocess
    from pathlib import Path

    T, H, W = mask_video.shape
    tmp = output_path + ".tmp.avi"

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (W, H), isColor=False)
    for frame in mask_video:
        writer.write(frame)
    writer.release()

    # Convert to H.264 with lossless quality
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp,
        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
        output_path
    ], check=True, capture_output=True)

    Path(tmp).unlink(missing_ok=True)
    return output_path


def save_mask_overlay(
    video_path: str, mask_video: np.ndarray, output_path: str, frame_idx: int = 0
) -> str:
    """Save a debug overlay image showing the mask on the video frame."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return ""

    mask = mask_video[frame_idx]
    # Resize mask to frame if needed
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    overlay = frame.copy()
    overlay[mask == 0] = [0, 0, 255]  # Red where removal will happen
    blended = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
    cv2.imwrite(output_path, blended)
    return output_path
