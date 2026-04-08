"""Display detected people and let user select the principal dancer."""
import cv2
import numpy as np
from clear_stage.detect_people import extract_frame

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0),
]


def draw_detections(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw numbered bounding boxes on frame. Returns annotated copy."""
    annotated = frame_bgr.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"#{i} ({det['score']:.2f})"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return annotated


def save_detection_preview(
    video_path: str, detections: list[dict], output_path: str, frame_idx: int = 0
) -> str:
    """Save annotated frame showing all detected people. Returns output path."""
    frame = extract_frame(video_path, frame_idx)
    annotated = draw_detections(frame, detections)
    cv2.imwrite(output_path, annotated)
    return output_path


def select_principal_cli(detections: list[dict]) -> int:
    """Prompt user to select which person to keep. Returns index."""
    print(f"\nDetected {len(detections)} people:")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        print(f"  #{i}: bbox={det['bbox']}, size={x2-x1}x{y2-y1}, conf={det['score']:.2f}")

    while True:
        try:
            choice = int(input(f"\nEnter the number of the person to KEEP [0-{len(detections)-1}]: "))
            if 0 <= choice < len(detections):
                return choice
        except ValueError:
            pass
        print(f"Please enter a number between 0 and {len(detections)-1}")
