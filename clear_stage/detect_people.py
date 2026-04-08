"""Detect all people in a video frame using GroundingDINO."""
import cv2
import numpy as np


def extract_frame(video_path: str, frame_idx: int = 0) -> np.ndarray:
    """Extract a single frame from a video file. Returns BGR numpy array."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def get_frame_count(video_path: str) -> int:
    """Get total number of frames in a video."""
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def detect_people(
    video_path: str,
    frame_idx: int = 0,
    confidence_threshold: float = 0.3,
    text_prompt: str = "person",
) -> list[dict]:
    """
    Detect all people in a single video frame.

    Returns list of dicts sorted left-to-right:
        [{"bbox": [x1, y1, x2, y2], "score": float}, ...]
    Coordinates are in pixel space of the original frame.
    """
    from groundingdino.util.inference import load_model, predict
    from PIL import Image

    frame_bgr = extract_frame(video_path, frame_idx)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    model = load_model(
        "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "groundingdino_swint_ogc.pth",
    )

    image_pil = Image.fromarray(frame_rgb)
    boxes, logits, phrases = predict(
        model=model,
        image=image_pil,
        caption=text_prompt,
        box_threshold=confidence_threshold,
        text_threshold=confidence_threshold,
    )

    results = []
    for box, score in zip(boxes, logits):
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        results.append({"bbox": [x1, y1, x2, y2], "score": float(score)})

    results.sort(key=lambda r: r["bbox"][0])
    return results


def find_good_detection_frame(
    video_path: str,
    sample_frames: list[int] | None = None,
    min_people: int = 2,
    confidence_threshold: float = 0.3,
) -> tuple[int, list[dict]]:
    """
    Try multiple frames to find one with good person detections.

    Useful when frame 0 shows a hand setting up the camera.
    Returns (frame_idx, detections).
    """
    total = get_frame_count(video_path)
    if sample_frames is None:
        # Try frame 0, then 5s in, 10s in, 15s in
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        sample_frames = [0, int(5 * fps), int(10 * fps), int(15 * fps)]
        sample_frames = [f for f in sample_frames if f < total]

    best_frame = 0
    best_detections = []

    for frame_idx in sample_frames:
        detections = detect_people(video_path, frame_idx, confidence_threshold)
        if len(detections) >= min_people:
            return frame_idx, detections
        if len(detections) > len(best_detections):
            best_frame = frame_idx
            best_detections = detections

    return best_frame, best_detections
