"""Convert person detections into config_points.json for VLM-MASK-REASONER."""
import json
from pathlib import Path


def bbox_to_points(bbox: list[int]) -> list[list[int]]:
    """Generate SAM2 click points from a bounding box.

    For tall bounding boxes (height > 200px), generates 3 points:
    center, upper body (25% from top), lower body (75% from top).
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    h = y2 - y1
    points = [[cx, cy]]
    if h > 200:
        points.append([cx, y1 + h // 4])
        points.append([cx, y1 + 3 * h // 4])
    return points


def generate_config(
    video_path: str, output_dir: str, detections: list[dict],
    principal_idx: int, frame_idx: int = 0,
    instruction: str = "remove the background people",
) -> dict:
    """Generate config_points.json. All people except principal get point prompts."""
    all_points = []
    for i, det in enumerate(detections):
        if i == principal_idx:
            continue
        all_points.extend(bbox_to_points(det["bbox"]))

    return {
        "videos": [{
            "video_path": str(Path(video_path).resolve()),
            "output_dir": str(Path(output_dir).resolve()),
            "instruction": instruction,
            "primary_points_by_frame": {str(frame_idx): all_points},
        }]
    }


def save_config(config: dict, output_path: str) -> str:
    """Write config to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path
