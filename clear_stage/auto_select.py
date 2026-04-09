"""Auto-select the principal dancer using multi-frame analysis.

Scores each detected person on centrality, motion, and size across
multiple sampled frames, then picks the highest-scoring person.
"""
import cv2
import numpy as np
from clear_stage.detect_people import detect_people, extract_frame, get_frame_count


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _bbox_area(bbox: list[int]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _iou(a: list[int], b: list[int]) -> float:
    """Intersection over union of two bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _match_detections(prev_dets: list[dict], curr_dets: list[dict],
                      iou_threshold: float = 0.1) -> dict[int, int]:
    """Match people across frames by IoU. Returns {curr_idx: prev_idx}."""
    matches = {}
    used_prev = set()
    for ci, cd in enumerate(curr_dets):
        best_iou, best_pi = 0, -1
        for pi, pd in enumerate(prev_dets):
            if pi in used_prev:
                continue
            score = _iou(cd["bbox"], pd["bbox"])
            if score > best_iou:
                best_iou, best_pi = score, pi
        if best_iou >= iou_threshold and best_pi >= 0:
            matches[ci] = best_pi
            used_prev.add(best_pi)
    return matches


def auto_select_principal(
    video_path: str,
    num_samples: int = 8,
    confidence_threshold: float = 0.3,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> tuple[int, list[dict], int]:
    """Auto-detect the principal dancer across multiple frames.

    Scores each person on: centrality (distance from frame center),
    motion (bounding box displacement), and size (bounding box area).

    Args:
        video_path: Path to video file
        num_samples: Number of frames to sample across the video
        confidence_threshold: GroundingDINO detection threshold
        weights: (centrality_weight, motion_weight, size_weight)

    Returns:
        (principal_index, detections_at_best_frame, best_frame_idx)
        where principal_index is into the detections list
    """
    total_frames = get_frame_count(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    frame_center = (frame_w / 2, frame_h / 2)
    max_dist = (frame_w**2 + frame_h**2) ** 0.5 / 2  # max possible distance from center

    # Sample frames spread across the video (skip first 5% and last 5%)
    start = int(total_frames * 0.05)
    end = int(total_frames * 0.95)
    sample_indices = [int(start + i * (end - start) / (num_samples - 1))
                      for i in range(num_samples)]

    # Detect people in each sampled frame
    print(f"Sampling {num_samples} frames for principal detection...")
    frame_detections = []
    for fidx in sample_indices:
        dets = detect_people(video_path, fidx, confidence_threshold)
        frame_detections.append((fidx, dets))
        print(f"  Frame {fidx}: {len(dets)} people")

    # Find the frame with the most people (best for initial matching)
    best_frame_idx, best_dets = max(frame_detections, key=lambda x: len(x[1]))
    if not best_dets:
        return 0, [], 0

    num_people = len(best_dets)

    # Track each person across frames
    # person_tracks[i] = list of (frame_idx, bbox) for person i in best_dets
    person_tracks = [[(best_frame_idx, d["bbox"])] for d in best_dets]

    # Match forward and backward from best frame
    for fidx, dets in frame_detections:
        if fidx == best_frame_idx:
            continue
        matches = _match_detections(best_dets, dets)
        for curr_i, best_i in matches.items():
            person_tracks[best_i].append((fidx, dets[curr_i]["bbox"]))

    # Score each person
    w_central, w_motion, w_size = weights
    scores = []

    for person_i, track in enumerate(person_tracks):
        # Centrality: average distance from frame center (lower = more central)
        centralities = []
        for _, bbox in track:
            cx, cy = _bbox_center(bbox)
            dist = ((cx - frame_center[0])**2 + (cy - frame_center[1])**2) ** 0.5
            centralities.append(1.0 - dist / max_dist)  # 1 = center, 0 = corner
        avg_centrality = np.mean(centralities)

        # Motion: total displacement across frames
        total_motion = 0
        sorted_track = sorted(track, key=lambda t: t[0])
        for i in range(1, len(sorted_track)):
            prev_cx, prev_cy = _bbox_center(sorted_track[i-1][1])
            curr_cx, curr_cy = _bbox_center(sorted_track[i][1])
            total_motion += ((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2) ** 0.5

        # Size: average bounding box area
        avg_area = np.mean([_bbox_area(bbox) for _, bbox in track])

        # Presence: how many frames this person appears in (bonus for consistency)
        presence = len(track) / len(frame_detections)

        scores.append({
            "person_idx": person_i,
            "centrality": avg_centrality,
            "motion": total_motion,
            "avg_area": avg_area,
            "presence": presence,
        })

    # Normalize motion and size to 0-1
    max_motion = max(s["motion"] for s in scores) or 1
    max_area = max(s["avg_area"] for s in scores) or 1

    for s in scores:
        s["motion_norm"] = s["motion"] / max_motion
        s["size_norm"] = s["avg_area"] / max_area
        s["total_score"] = (
            w_central * s["centrality"]
            + w_motion * s["motion_norm"]
            + w_size * s["size_norm"]
        )
        # Bonus for appearing in most frames
        s["total_score"] *= (0.5 + 0.5 * s["presence"])

    # Sort by score
    scores.sort(key=lambda s: s["total_score"], reverse=True)
    principal_idx = scores[0]["person_idx"]

    print(f"\nPrincipal detection scores:")
    for s in scores:
        marker = " <-- PRINCIPAL" if s["person_idx"] == principal_idx else ""
        print(f"  #{s['person_idx']}: centrality={s['centrality']:.2f} "
              f"motion={s['motion_norm']:.2f} size={s['size_norm']:.2f} "
              f"presence={s['presence']:.1f} "
              f"TOTAL={s['total_score']:.3f}{marker}")

    return principal_idx, best_dets, best_frame_idx
