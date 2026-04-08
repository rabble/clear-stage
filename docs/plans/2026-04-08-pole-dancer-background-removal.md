# Pole Dance Studio Background Person Removal — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that removes background dancers from pole studio videos, keeping only the principal dancer and their reflections, using GroundingDINO + SAM 2 for detection/masking and VOID for inpainting.

**Architecture:** A Python CLI script that automates person detection (GroundingDINO), generates SAM 2 point prompts for all non-principal people, runs the existing VLM-MASK-REASONER pipeline (Stages 2-4), chunks the video for parallel VOID inference, and stitches results. The void-model repo is used as a submodule dependency. Runs on RunPod A100 instances.

**Tech Stack:** Python, GroundingDINO, SAM 2, Gemini API, VOID (CogVideoX), ffmpeg, RunPod

---

## File Structure

```
clear-stage/
├── docs/
│   ├── specs/           # Design spec
│   └── plans/           # This plan
├── clear_stage/
│   ├── __init__.py
│   ├── detect_people.py          # GroundingDINO person detection on frame 1
│   ├── select_principal.py       # Display detections, user picks who to keep
│   ├── generate_points_config.py # Convert detections → config_points.json for SAM 2
│   ├── chunk_video.py            # Split video into 197-frame chunks with overlap
│   ├── stitch_chunks.py          # Reassemble chunks with crossfade blending
│   └── run_pipeline.py           # Orchestrate full end-to-end pipeline
├── tests/
│   ├── test_detect_people.py
│   ├── test_chunk_video.py
│   └── test_generate_config.py
├── void-model/                   # git submodule → netflix/void-model
├── requirements.txt
└── README.md
```

All new code lives in `clear_stage/`. It calls into `void-model/VLM-MASK-REASONER/` and `void-model/inference/` without modifying them.

---

## Chunk 1: Project Setup & Person Detection

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `clear_stage/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Add void-model as git submodule**

```bash
cd /Users/rabble/code/personal/clear-stage
git submodule add https://github.com/Netflix/void-model.git void-model
```

- [ ] **Step 2: Create requirements.txt**

```
# Person detection
groundingdino-py

# Segmentation (SAM2 installed from void-model's requirements)
# pip install -e void-model/  handles this

# Video processing
opencv-python>=4.8
ffmpeg-python

# VLM
google-generativeai

# Utilities
numpy
torch
Pillow
```

Check that `groundingdino-py` is the correct package name in April 2026. If not, use `pip install git+https://github.com/IDEA-Research/GroundingDINO.git`.

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.env
*.mp4
*.avi
*.safetensors
*.pt
*.pth
/work/
/output/
```

- [ ] **Step 4: Create package init**

```python
# clear_stage/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements.txt clear_stage/__init__.py .gitmodules void-model
git commit -m "feat: project scaffolding with void-model submodule"
```

---

### Task 2: Person Detection with GroundingDINO

**Files:**
- Create: `clear_stage/detect_people.py`
- Create: `tests/test_detect_people.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_detect_people.py
import pytest
from pathlib import Path


def test_detect_people_returns_bounding_boxes():
    """detect_people should return a list of dicts with bbox and score."""
    from clear_stage.detect_people import detect_people

    video_path = Path("void-model/sample/lime/input_video.mp4")
    if not video_path.exists():
        pytest.skip("Sample video not available")

    results = detect_people(str(video_path), frame_idx=0)

    assert isinstance(results, list)
    for r in results:
        assert "bbox" in r
        assert "score" in r
        assert len(r["bbox"]) == 4
        assert all(isinstance(v, (int, float)) for v in r["bbox"])
        assert 0.0 <= r["score"] <= 1.0


def test_detect_people_sorted_left_to_right():
    """Results should be sorted by x-position for consistent numbering."""
    from clear_stage.detect_people import detect_people

    video_path = Path("void-model/sample/lime/input_video.mp4")
    if not video_path.exists():
        pytest.skip("Sample video not available")

    results = detect_people(str(video_path), frame_idx=0)
    if len(results) >= 2:
        for i in range(len(results) - 1):
            assert results[i]["bbox"][0] <= results[i + 1]["bbox"][0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_detect_people.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement detect_people**

```python
# clear_stage/detect_people.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_detect_people.py -v
```

- [ ] **Step 5: Commit**

```bash
git add clear_stage/detect_people.py tests/test_detect_people.py
git commit -m "feat: add person detection with GroundingDINO"
```

---

### Task 3: Principal Selection (CLI)

**Files:**
- Create: `clear_stage/select_principal.py`

- [ ] **Step 1: Implement selection with preview image**

```python
# clear_stage/select_principal.py
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
```

- [ ] **Step 2: Commit**

```bash
git add clear_stage/select_principal.py
git commit -m "feat: add principal dancer selection with CLI and preview"
```

---

### Task 4: Generate config_points.json for SAM 2

**Files:**
- Create: `clear_stage/generate_points_config.py`
- Create: `tests/test_generate_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_generate_config.py
def test_generate_config_excludes_principal():
    """Config should contain points for all people EXCEPT the principal."""
    from clear_stage.generate_points_config import generate_config

    detections = [
        {"bbox": [100, 100, 200, 400], "score": 0.9},
        {"bbox": [400, 100, 500, 400], "score": 0.8},  # principal
        {"bbox": [700, 100, 800, 400], "score": 0.7},
    ]

    config = generate_config(
        video_path="test.mp4", output_dir="out", detections=detections, principal_idx=1,
    )

    entry = config["videos"][0]
    points = entry["primary_points_by_frame"]["0"]
    # Points from person 0 and 2 only — person 1 excluded
    assert len(points) >= 2
    # No point should be inside person 1's bbox (400-500, 100-400)
    for p in points:
        assert not (400 <= p[0] <= 500 and 100 <= p[1] <= 400)


def test_bbox_to_points_tall_person():
    """Tall bounding boxes get multiple points."""
    from clear_stage.generate_points_config import bbox_to_points

    points = bbox_to_points([100, 50, 300, 600])
    assert len(points) >= 2


def test_bbox_to_points_small_person():
    """Small bounding boxes get one center point."""
    from clear_stage.generate_points_config import bbox_to_points

    points = bbox_to_points([100, 100, 200, 250])
    assert len(points) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_generate_config.py -v
```

- [ ] **Step 3: Implement config generation**

```python
# clear_stage/generate_points_config.py
"""Convert person detections into config_points.json for VLM-MASK-REASONER."""
import json
from pathlib import Path


def bbox_to_points(bbox: list[int]) -> list[list[int]]:
    """Generate SAM2 click points from a bounding box."""
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
    principal_idx: int, instruction: str = "remove the background people",
) -> dict:
    """Generate config_points.json. All people except principal get point prompts."""
    all_points = []
    for i, det in enumerate(detections):
        if i == principal_idx:
            continue
        all_points.extend(bbox_to_points(det["bbox"]))

    return {
        "videos": [{
            "video_path": video_path,
            "output_dir": output_dir,
            "instruction": instruction,
            "primary_points_by_frame": {"0": all_points},
        }]
    }


def save_config(config: dict, output_path: str) -> str:
    """Write config to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_generate_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add clear_stage/generate_points_config.py tests/test_generate_config.py
git commit -m "feat: generate SAM2 config from person detections"
```

---

## Chunk 2: Video Chunking & Stitching

### Task 5: Video Chunking

**Files:**
- Create: `clear_stage/chunk_video.py`
- Create: `tests/test_chunk_video.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chunk_video.py
def test_calculate_chunks_short_video():
    from clear_stage.chunk_video import calculate_chunks
    chunks = calculate_chunks(total_frames=100, max_frames=197, overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == (0, 100)


def test_calculate_chunks_exact_boundary():
    from clear_stage.chunk_video import calculate_chunks
    chunks = calculate_chunks(total_frames=197, max_frames=197, overlap=20)
    assert len(chunks) == 1


def test_calculate_chunks_needs_splitting():
    from clear_stage.chunk_video import calculate_chunks
    chunks = calculate_chunks(total_frames=400, max_frames=197, overlap=20)
    assert len(chunks) >= 2
    assert chunks[0] == (0, 197)
    assert chunks[1][0] == 177  # 197 - 20 overlap
    assert chunks[-1][1] >= 400
    for i in range(len(chunks) - 1):
        assert chunks[i][1] - chunks[i + 1][0] == 20


def test_downsample_frames():
    from clear_stage.chunk_video import calculate_downsampled_frames
    assert calculate_downsampled_frames(900, 30, 12) == 360
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_chunk_video.py -v
```

- [ ] **Step 3: Implement chunking**

```python
# clear_stage/chunk_video.py
"""Split video into chunks for parallel VOID processing."""
import subprocess
import json
from pathlib import Path


def get_video_info(video_path: str) -> dict:
    """Get video metadata via ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    vs = next(s for s in data["streams"] if s["codec_type"] == "video")
    fps_n, fps_d = vs["r_frame_rate"].split("/")
    fps = float(fps_n) / float(fps_d)
    total = int(vs.get("nb_frames", 0))
    if total == 0:
        total = int(float(data["format"]["duration"]) * fps)
    return {"fps": fps, "total_frames": total,
            "width": int(vs["width"]), "height": int(vs["height"]),
            "duration": float(data["format"]["duration"])}


def calculate_downsampled_frames(total_frames: int, source_fps: float, target_fps: float) -> int:
    return int(total_frames * target_fps / source_fps)


def calculate_chunks(total_frames: int, max_frames: int = 197, overlap: int = 20) -> list[tuple[int, int]]:
    """Calculate chunk boundaries (start, end) with overlap. end is exclusive."""
    if total_frames <= max_frames:
        return [(0, total_frames)]
    chunks = []
    start = 0
    stride = max_frames - overlap
    while start < total_frames:
        end = min(start + max_frames, total_frames)
        chunks.append((start, end))
        if end >= total_frames:
            break
        start += stride
    return chunks


def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio track. Returns True if audio exists."""
    r = subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", output_path],
                       capture_output=True)
    return r.returncode == 0


def downsample_video(video_path: str, output_path: str, target_fps: float = 12.0) -> str:
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-r", str(target_fps),
                    "-c:v", "libx264", "-crf", "18", "-an", output_path],
                   check=True, capture_output=True)
    return output_path


def split_video_chunk(video_path: str, start: int, end: int, fps: float, output_path: str) -> str:
    start_t, dur = start / fps, (end - start) / fps
    subprocess.run(["ffmpeg", "-y", "-ss", str(start_t), "-i", video_path,
                    "-t", str(dur), "-c:v", "libx264", "-crf", "18", "-an", output_path],
                   check=True, capture_output=True)
    return output_path


def prepare_chunks(
    video_path: str, mask_path: str, prompt: str, output_dir: str,
    target_fps: float = 12.0, max_frames: int = 197, overlap: int = 20,
) -> list[dict]:
    """Downsample, split video+mask, write prompt.json per chunk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds_video = str(out / "downsampled_video.mp4")
    ds_mask = str(out / "downsampled_mask.mp4")
    downsample_video(video_path, ds_video, target_fps)
    downsample_video(mask_path, ds_mask, target_fps)
    extract_audio(video_path, str(out / "audio.aac"))

    ds_info = get_video_info(ds_video)
    chunks = calculate_chunks(ds_info["total_frames"], max_frames, overlap)

    chunk_infos = []
    for i, (start, end) in enumerate(chunks):
        cd = out / f"chunk_{i:03d}"
        cd.mkdir(exist_ok=True)
        split_video_chunk(ds_video, start, end, target_fps, str(cd / "input_video.mp4"))
        split_video_chunk(ds_mask, start, end, target_fps, str(cd / "quadmask_0.mp4"))
        with open(cd / "prompt.json", "w") as f:
            json.dump({"bg": prompt}, f)
        chunk_infos.append({"chunk_dir": str(cd), "chunk_name": f"chunk_{i:03d}",
                           "start": start, "end": end})
    return chunk_infos
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_chunk_video.py -v
```

- [ ] **Step 5: Commit**

```bash
git add clear_stage/chunk_video.py tests/test_chunk_video.py
git commit -m "feat: video chunking with overlap for parallel VOID processing"
```

---

### Task 6: Chunk Stitching

**Files:**
- Create: `clear_stage/stitch_chunks.py`

- [ ] **Step 1: Implement stitching with crossfade**

```python
# clear_stage/stitch_chunks.py
"""Stitch processed video chunks with crossfade blending."""
import subprocess
import cv2
import numpy as np
from pathlib import Path


def read_video_frames(video_path: str) -> np.ndarray:
    """Read all frames. Returns (T, H, W, 3) uint8."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.stack(frames) if frames else np.array([])


def write_video_frames(frames: np.ndarray, output_path: str, fps: float = 12.0) -> str:
    """Write frames to H.264 video."""
    h, w = frames.shape[1], frames.shape[2]
    tmp = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-crf", "18", output_path],
                   check=True, capture_output=True)
    Path(tmp).unlink()
    return output_path


def stitch_chunks(
    chunk_infos: list[dict], chunk_output_dir: str, output_path: str,
    overlap: int = 20, fps: float = 12.0,
) -> str:
    """Stitch processed chunks with linear crossfade in overlap regions."""
    from clear_stage.run_pipeline import find_void_output

    if len(chunk_infos) == 1:
        src = find_void_output(chunk_output_dir, chunk_infos[0]["chunk_name"])
        subprocess.run(["cp", src, output_path], check=True)
        return output_path

    chunk_frames = []
    for info in chunk_infos:
        vp = find_void_output(chunk_output_dir, info["chunk_name"])
        chunk_frames.append(read_video_frames(vp))

    result = list(chunk_frames[0][:-overlap])
    for i in range(1, len(chunk_frames)):
        prev, curr = chunk_frames[i - 1], chunk_frames[i]
        for j in range(overlap):
            alpha = j / overlap
            blended = ((1 - alpha) * prev[len(prev) - overlap + j].astype(float)
                       + alpha * curr[j].astype(float)).astype(np.uint8)
            result.append(blended)
        tail = curr[overlap:] if i == len(chunk_frames) - 1 else curr[overlap:-overlap]
        result.extend(tail)

    return write_video_frames(np.stack(result), output_path, fps)


def remux_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """Re-attach audio track to processed video."""
    if not Path(audio_path).exists():
        subprocess.run(["cp", video_path, output_path], check=True)
        return output_path
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
                    "-c:v", "copy", "-c:a", "aac", "-shortest", output_path],
                   check=True, capture_output=True)
    return output_path
```

- [ ] **Step 2: Commit**

```bash
git add clear_stage/stitch_chunks.py
git commit -m "feat: chunk stitching with crossfade and audio remux"
```

---

## Chunk 3: Pipeline Orchestration & Testing

### Task 7: End-to-End Pipeline Orchestrator

**Files:**
- Create: `clear_stage/run_pipeline.py`

- [ ] **Step 1: Implement orchestrator**

```python
# clear_stage/run_pipeline.py
"""
End-to-end pipeline: detect → select → mask → chunk → VOID → stitch.

Usage:
    python -m clear_stage.run_pipeline \
        --video input.mp4 \
        --output output.mp4 \
        --prompt "A pole dance studio with mirrors and wooden floor" \
        --work-dir /tmp/clear_stage_work
"""
import argparse
import subprocess
import sys
from pathlib import Path

from clear_stage.detect_people import detect_people
from clear_stage.select_principal import select_principal_cli, save_detection_preview
from clear_stage.generate_points_config import generate_config, save_config
from clear_stage.chunk_video import prepare_chunks, extract_audio
from clear_stage.stitch_chunks import stitch_chunks, remux_audio


def run_mask_generation(config_path: str, device: str = "cuda") -> None:
    """Run VLM-MASK-REASONER stages 1-4."""
    repo_root = Path(__file__).parent.parent
    void_root = repo_root / "void-model"
    script = void_root / "VLM-MASK-REASONER" / "run_pipeline.sh"
    sam2_ckpt = void_root / "sam2_hiera_large.pt"
    subprocess.run(
        ["bash", str(script), config_path,
         "--sam2-checkpoint", str(sam2_ckpt),
         "--device", device],
        cwd=str(void_root),  # run_pipeline.sh uses relative paths
        check=True,
    )


def find_void_output(chunk_dir: str, chunk_name: str) -> str:
    """Find VOID output file by glob pattern (output naming varies)."""
    import glob
    # predict_v2v.py saves to config.experiment.save_path with pattern:
    # {seq_name}-fg=_01-0001.mp4 or similar
    patterns = [
        f"{chunk_dir}/{chunk_name}/**/output*.mp4",
        f"{chunk_dir}/{chunk_name}/**/*-fg=*.mp4",
        f"{chunk_dir}/**/{chunk_name}*.mp4",
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No VOID output found for chunk {chunk_name} in {chunk_dir}")


def run_void_inference(
    data_rootdir: str, seq_names: list[str],
    transformer_path: str = "void-model/void_pass1.safetensors",
    save_path: str = None,
) -> None:
    """Run VOID Pass 1 inference.

    NOTE: predict_v2v.py uses ml_collections config flags (Python .py files),
    NOT JSON config files. The --config flag expects a Python config module.
    """
    repo_root = Path(__file__).parent.parent
    script = repo_root / "void-model" / "inference" / "cogvideox_fun" / "predict_v2v.py"
    py_config = str(repo_root / "void-model" / "config" / "quadmask_cogvideox.py")

    if save_path is None:
        save_path = str(Path(data_rootdir) / "void_output")

    cmd = [
        "python", str(script),
        f"--config={py_config}",  # ml_collections expects --config=path.py format
        f"--config.data.data_rootdir={data_rootdir}",
        f"--config.experiment.run_seqs={','.join(seq_names)}",
        f"--config.video_model.transformer_path={str(repo_root / transformer_path)}",
        f"--config.experiment.save_path={save_path}",
    ]
    subprocess.run(cmd, cwd=str(repo_root / "void-model"), check=True)


def main():
    parser = argparse.ArgumentParser(description="Remove background dancers from pole dance video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True, help="Background description after removal")
    parser.add_argument("--work-dir", default="/tmp/clear_stage_work")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transformer-path", default="void-model/void_pass1.safetensors")
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--max-frames", type=int, default=197)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--principal", type=int, default=None,
                        help="Skip interactive selection, use this index")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Step 1: Detect people
    print("\n=== Step 1: Detecting people ===")
    detections = detect_people(args.video)
    if not detections:
        print("No people detected. Nothing to remove.")
        subprocess.run(["cp", args.video, args.output], check=True)
        return
    print(f"Found {len(detections)} people")

    # Step 2: Select principal
    preview = str(work / "detection_preview.jpg")
    save_detection_preview(args.video, detections, preview)
    print(f"Preview saved: {preview}")

    if args.principal is not None:
        principal_idx = args.principal
    else:
        principal_idx = select_principal_cli(detections)

    if len(detections) == 1:
        print("Only one person detected (the principal). Nothing to remove.")
        subprocess.run(["cp", args.video, args.output], check=True)
        return

    # Step 3: Generate masks
    print("\n=== Step 3: Generating masks ===")
    mask_dir = str(work / "masks")
    config = generate_config(args.video, mask_dir, detections, principal_idx)
    config_path = str(work / "config_points.json")
    save_config(config, config_path)
    run_mask_generation(config_path, args.device)

    # Step 4: Chunk
    print("\n=== Step 4: Chunking ===")
    quadmask = str(Path(mask_dir) / "quadmask_0.mp4")
    chunks_dir = str(work / "chunks")
    chunk_infos = prepare_chunks(
        args.video, quadmask, args.prompt, chunks_dir,
        args.target_fps, args.max_frames, args.overlap,
    )
    print(f"Split into {len(chunk_infos)} chunks")

    # Step 5: VOID inference (serial for now — parallelize on RunPod later)
    print("\n=== Step 5: VOID inference ===")
    run_void_inference(
        chunks_dir, [c["chunk_name"] for c in chunk_infos],
        args.transformer_path,
    )

    # Step 6: Stitch
    print("\n=== Step 6: Stitching ===")
    stitched = str(work / "stitched.mp4")
    stitch_chunks(chunk_infos, chunks_dir, stitched, args.overlap, args.target_fps)

    # Step 7: Audio
    print("\n=== Step 7: Re-attaching audio ===")
    audio = str(Path(chunks_dir) / "audio.aac")
    remux_audio(stitched, audio, args.output)
    print(f"\nDone! Output: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add clear_stage/run_pipeline.py
git commit -m "feat: end-to-end pipeline orchestrator"
```

---

### Task 8: Test End-to-End with Sample Video

- [ ] **Step 1: Test detection on sample video**

```bash
python -c "
from clear_stage.detect_people import detect_people
results = detect_people('void-model/sample/lime/input_video.mp4')
print(f'Detected {len(results)} objects')
for i, r in enumerate(results): print(f'  #{i}: {r}')
"
```

Validates GroundingDINO works. The lime sample may not have people — that's fine.

- [ ] **Step 2: Test full pipeline on a real pole dance video (short clip)**

```bash
python -m clear_stage.run_pipeline \
    --video /path/to/short_clip.mp4 \
    --output /tmp/pole_output.mp4 \
    --prompt "A pole dance studio with mirrors and wooden floor" \
    --work-dir /tmp/clear_stage_test
```

Start with a 10-15 second clip for fast iteration.

- [ ] **Step 3: Review and document results**

Check: background people removed? Principal intact? Mirror reflections correct? Chunk boundaries visible? Document findings.

- [ ] **Step 4: Commit findings and fixes**

```bash
git add -A
git commit -m "test: end-to-end validation and fixes"
```

---

## Implementation Notes

### RunPod Parallelization (future task)

Replace the serial VOID loop in Task 7 with:
1. Upload chunks to shared storage (GCS or RunPod volume)
2. Spin up N RunPod serverless workers
3. Each worker processes one chunk
4. Collect results, stitch locally

### SAM 3.1 Upgrade (future task)

Once pipeline works with GroundingDINO + SAM 2:
1. Install SAM 3.1: `pip install git+https://github.com/facebookresearch/sam3.git`
2. Test text-prompt detection and video tracking
3. Replace `detect_people.py` and Stage 1 with SAM 3.1
4. Remove GroundingDINO dependency

### Model Checkpoint Download

Before running, you must download model weights:
```bash
cd void-model
# VOID Pass 1 checkpoint
hf download netflix/void-model void_pass1.safetensors --local-dir .
# Base CogVideoX model
hf download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP --local-dir ./CogVideoX-Fun-V1.5-5b-InP
# SAM2 checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
# GroundingDINO weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Preview Tier (deferred from spec)

Add `--quality preview` flag to run_pipeline.py:
- Resolution: 192x336 instead of 384x672
- Inference steps: 15 instead of 50
- ~10x faster and cheaper

### RIFE Frame Interpolation (deferred from spec)

V1 outputs at 12fps. RIFE upsampling to 30fps is deferred. If 12fps is unacceptable, process at native fps (more chunks, higher cost) rather than interpolating.

### Web UI (deferred from spec)

The spec describes a mobile-friendly web upload interface. V1 is CLI only. Web UI will be added once the pipeline is validated end-to-end.
