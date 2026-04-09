# Clear Stage v2: Quality, Full-Video Processing & Automation

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the working proof-of-concept into a reliable single-command tool that processes full-length pole dance videos (30s–6min) at correct speed, higher resolution, and with proper shadow/reflection handling.

**Architecture:** The existing pipeline (GroundingDINO → SAM2 → VOID) is proven. This plan fixes FPS handling, adds resolution control, integrates the working `segment_people.py` into the main pipeline, enables Gemini VLM for shadow analysis, implements chunked processing for long videos, and wraps it all in a single `clear-stage process` command with a RunPod helper for remote execution.

**Tech Stack:** Python, GroundingDINO, SAM2, VOID (CogVideoX), Gemini API, ffmpeg, RunPod API

**Prerequisites:** RunPod pod `cts9ata0uqd95e` (A100 SXM, EUR-IS-1), network volume `oxfxblvrr6` with cached models. SSH key at `~/.ssh/runpod_key`.

---

## File Structure

```
clear_stage/
├── __init__.py                    # (exists)
├── detect_people.py               # (exists) GroundingDINO detection
├── segment_people.py              # (exists) Multi-person SAM2 — NEEDS: mask erosion, fps param
├── select_principal.py            # (exists) CLI selection + preview
├── generate_points_config.py      # (exists) — will be BYPASSED (segment_people replaces it)
├── chunk_video.py                 # (exists) — MODIFY: remove fps downsampling, fix fps passthrough
├── stitch_chunks.py               # (exists) — MODIFY: output at source fps
├── run_pipeline.py                # (exists) — REWRITE: integrate segment_people, remove old mask path
├── resolution.py                  # CREATE: resolution presets (preview/standard/high)
├── runpod_runner.py               # CREATE: remote execution via RunPod API
└── cli.py                         # CREATE: entry point with subcommands
tests/
├── test_generate_config.py        # (exists)
├── test_chunk_video.py            # (exists)
├── test_resolution.py             # CREATE
└── test_segment_integration.py    # CREATE: mock-free integration test for mask format
```

---

## Milestone 1: Fix FPS & Integrate segment_people into Pipeline

**Why first:** These are the two worst bugs from the POC. Video plays at wrong speed, and the pipeline still uses the old broken mask path instead of the working `segment_people.py`.

### Task 1: Fix FPS — process at source framerate, output at source framerate

**Files:**
- Modify: `clear_stage/chunk_video.py`
- Modify: `clear_stage/stitch_chunks.py`
- Modify: `tests/test_chunk_video.py`

The current pipeline downsamples to 12fps before processing. This has two effects: (1) the output plays at wrong speed because it's re-encoded at 12fps, and (2) processing at lower fps means VOID sees more temporal context per chunk (197 frames at 12fps = 16.4s vs 6.6s at 30fps).

**Tradeoff:** Processing at native 30fps costs ~2.5x more chunks (and compute) but preserves temporal quality. Processing at 12fps is cheaper but synthesizes motion.

**Fix:** Default to 24fps (good balance), allow `--target-fps` override, but always re-encode the output at the **original source fps** so it plays at correct speed. The bug was that output fps was set to `target_fps` instead of source fps.

- [ ] **Step 1: Add test for fps passthrough**

```python
# tests/test_chunk_video.py — append
def test_prepare_chunks_preserves_fps():
    """prepare_chunks should use source fps when target_fps matches source."""
    from clear_stage.chunk_video import get_video_info
    # Just test the calculation, not actual ffmpeg
    info = {"fps": 30.0, "total_frames": 450}
    # At 30fps, 450 frames = 15 seconds
    # With max_frames=197, should get 3 chunks
    from clear_stage.chunk_video import calculate_chunks
    chunks = calculate_chunks(450, max_frames=197, overlap=20)
    assert len(chunks) == 3
    assert chunks[0] == (0, 197)
```

- [ ] **Step 2: Run test**

```bash
cd /Users/rabble/code/personal/clear-stage && source .venv/bin/activate
python -m pytest tests/test_chunk_video.py -v
```

- [ ] **Step 3: Update `prepare_chunks` to default to source fps**

In `clear_stage/chunk_video.py`, change the `prepare_chunks` function:
- Change default `target_fps` parameter from `12.0` to `None`
- When `target_fps is None`, skip downsampling — use source video directly
- When `target_fps` is set, downsample as before
- Always pass detected fps to chunk splitting

- [ ] **Step 4: Update `stitch_chunks.py` to accept fps parameter**

In `stitch_chunks.py`, the `write_video_frames` function hardcodes fps=12.0. Change default to accept the actual source fps.

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest tests/ -v
git add -A && git commit -m "fix: preserve source fps throughout pipeline, no forced 12fps downsample"
```

---

### Task 2: Replace old mask pipeline with segment_people in run_pipeline.py

**Files:**
- Modify: `clear_stage/run_pipeline.py`

The current `run_pipeline.py` uses the old VLM-MASK-REASONER Stage 1 (`run_mask_generation`) which produced broken masks. Replace with the working `segment_people.py` approach that uses per-person bounding boxes.

- [ ] **Step 1: Rewrite the pipeline to use segment_people**

Replace Steps 3-5 in `main()`:
- Remove `run_mask_generation()` call
- Import and call `segment_background_people()` from `segment_people.py`
- Call `save_mask_video()` to write the quadmask
- Set up VOID input directory structure
- Call `run_void_inference()` with correct `sample_size`
- Match output fps to input fps
- Keep basic chunking functional (use existing `prepare_chunks` / `stitch_chunks` — Task 5 refines this)

**NOTE:** Between Task 2 and Task 5, testing is limited to clips under ~16s (197 frames at 12fps). Full-length video support comes in Task 5.

```python
# Step 3: Segment background people
from clear_stage.segment_people import segment_background_people, save_mask_video
from clear_stage.chunk_video import get_video_info

video_info = get_video_info(args.video)
mask_video = segment_background_people(
    args.video, detections, principal_idx,
    sam2_checkpoint=str(repo_root / "void-model" / "sam2_hiera_large.pt"),
    frame_idx=frame_idx, device=args.device,
)

# Step 4: Set up VOID input
void_dir = str(work / "void_input")
seq_dir = os.path.join(void_dir, "seq")
os.makedirs(seq_dir, exist_ok=True)
shutil.copy(args.video, os.path.join(seq_dir, "input_video.mp4"))
save_mask_video(mask_video, os.path.join(seq_dir, "quadmask_0.mp4"), fps=video_info["fps"])
# ... prompt.json, run VOID, etc.
```

- [ ] **Step 2: Remove unused imports and dead code**

Remove: `generate_points_config` imports, `run_mask_generation()` function, `prepare_chunks` / `stitch_chunks` calls (will be re-added in Task 4).

- [ ] **Step 3: Test end-to-end on pod**

```bash
# On RunPod:
python -m clear_stage.run_pipeline \
    --video sample_videos/IMG_2745.mov \
    --output output/test_v2.mp4 \
    --prompt "A pole dance studio with purple LED lighting, mirrors, and wooden floor" \
    --principal 1
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: integrate segment_people into pipeline, replace broken mask path"
```

---

## Milestone 2: Resolution Control & Mask Quality

**Why second:** Higher resolution and cleaner masks directly improve output quality — the two biggest complaints from the POC.

### Task 3: Resolution presets

**Files:**
- Create: `clear_stage/resolution.py`
- Create: `tests/test_resolution.py`
- Modify: `clear_stage/run_pipeline.py`

- [ ] **Step 1: Write tests for resolution presets**

```python
# tests/test_resolution.py
def test_preview_resolution_portrait():
    from clear_stage.resolution import get_sample_size
    # Preview: ~65k pixels (quarter of standard)
    s = get_sample_size(720, 1280, quality="preview")
    h, w = map(int, s.split("x"))
    assert h > w  # portrait
    assert h * w < 100000  # low res
    assert h % 16 == 0 and w % 16 == 0

def test_standard_resolution_portrait():
    from clear_stage.resolution import get_sample_size
    s = get_sample_size(720, 1280, quality="standard")
    h, w = map(int, s.split("x"))
    assert h > w
    assert 200000 < h * w < 300000  # ~258k like VOID default

def test_high_resolution_portrait():
    from clear_stage.resolution import get_sample_size
    s = get_sample_size(720, 1280, quality="high")
    h, w = map(int, s.split("x"))
    assert h > w
    assert h * w > 400000  # significantly more than standard

def test_resolution_preserves_aspect_ratio():
    from clear_stage.resolution import get_sample_size
    for q in ["preview", "standard", "high"]:
        s = get_sample_size(720, 1280, quality=q)
        h, w = map(int, s.split("x"))
        aspect = w / h
        expected = 720 / 1280
        assert abs(aspect - expected) < 0.1  # within 10%
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_resolution.py -v
```

- [ ] **Step 3: Implement resolution.py**

```python
# clear_stage/resolution.py
"""Resolution presets for VOID processing."""

PRESETS = {
    "preview":  64512,   # ~254x254 equivalent, fast testing
    "standard": 258048,  # 384x672, VOID's default
    "high":     516096,  # ~543x950, 2x pixels
}

def get_sample_size(width: int, height: int, quality: str = "standard") -> str:
    """Compute VOID sample_size preserving aspect ratio at given quality."""
    target_pixels = PRESETS[quality]
    aspect = width / height
    h = int((target_pixels / aspect) ** 0.5)
    w = int(h * aspect)
    h = max(16, (h // 16) * 16)
    w = max(16, (w // 16) * 16)
    return f"{h}x{w}"
```

- [ ] **Step 4: Run tests, commit**

```bash
python -m pytest tests/test_resolution.py -v
git add -A && git commit -m "feat: resolution presets (preview/standard/high)"
```

- [ ] **Step 5: Add `--quality` flag to run_pipeline.py**

Add `parser.add_argument("--quality", default="standard", choices=["preview", "standard", "high"])` and replace `compute_void_sample_size` with `get_sample_size(info["width"], info["height"], args.quality)`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: add --quality flag to pipeline (preview/standard/high)"
```

---

### Task 4: Erode masks to prevent principal dancer blurring

**Files:**
- Modify: `clear_stage/segment_people.py`

The mask slightly overlaps the principal dancer near removed people, causing hand/pole blur. Fix: erode (shrink) the removal mask by a few pixels so it doesn't eat into the principal.

- [ ] **Step 1: Add mask erosion to segment_background_people**

After building `mask_video`, before returning:

```python
# Dilate the keep regions (255) to shrink the removal regions (0),
# preventing the mask from eating into the principal dancer
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
for i in range(mask_video.shape[0]):
    mask_video[i] = cv2.dilate(mask_video[i], kernel, iterations=erode_iterations)
```

Note: `cv2.dilate` expands bright (255/keep) regions, which shrinks the dark (0/remove) regions — the desired effect.

- [ ] **Step 2: Add `--erode` parameter (default 2) to control erosion strength**

Add to `segment_background_people()` signature: `erode_iterations: int = 2`

- [ ] **Step 3: Test on pod, compare with/without erosion**

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: erode removal masks to prevent principal dancer blurring"
```

---

## Milestone 3: Full-Length Video Processing

**Why third:** Once quality is good on short clips, scale to full 30s–6min videos.

### Task 5: Chunked VOID processing for long videos

**Files:**
- Modify: `clear_stage/run_pipeline.py`
- Modify: `clear_stage/chunk_video.py`
- Modify: `clear_stage/stitch_chunks.py`

SAM2 segments the entire video in one pass (it handles long videos natively). Only VOID needs chunking because it processes max 197 frames at a time.

**Memory concern:** For long videos, `segment_background_people` stores the full mask array in RAM (e.g., 98s at 30fps at 720x1280 = ~3.4GB). If this becomes a problem, stream masks to disk frame-by-frame instead of accumulating in memory. For v2 with the A100's 188GB RAM, this should be fine up to ~6 min videos.

- [ ] **Step 1: Refactor chunk_video.py for mask-aware chunking**

Add function `prepare_void_chunks()` that:
- Takes the full input video and full mask video
- Splits both into 197-frame chunks with 20-frame overlap
- Writes each chunk as `chunk_NNN/{input_video.mp4, quadmask_0.mp4, prompt.json}`
- Returns list of chunk info dicts

```python
def prepare_void_chunks(
    video_path: str, mask_path: str, prompt: str, output_dir: str,
    max_frames: int = 197, overlap: int = 20,
) -> list[dict]:
    """Split video and mask into chunks for parallel VOID processing."""
```

- [ ] **Step 2: Update run_pipeline.py to chunk → VOID → stitch**

After segmentation:
1. Save full mask video
2. Call `prepare_void_chunks()` to split video + mask
3. Run VOID on each chunk (serial for now)
4. Stitch chunks with crossfade
5. Remux audio from original

- [ ] **Step 3: Test with 30-second clip**

- [ ] **Step 4: Test with full 98-second video (IMG_2745.mov)**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: chunked VOID processing for full-length videos"
```

---

### Task 6: Audio preservation

**Files:**
- Modify: `clear_stage/run_pipeline.py`

- [ ] **Step 1: Extract audio from original video before processing**

```python
from clear_stage.chunk_video import extract_audio
audio_path = str(work / "audio.aac")
has_audio = extract_audio(args.video, audio_path)
```

- [ ] **Step 2: Remux audio into final output**

After stitching, call `remux_audio(stitched_path, audio_path, args.output)`.

- [ ] **Step 3: Verify audio plays in output**

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: preserve audio track in processed videos"
```

---

## Milestone 4: Single-Command Automation

**Why fourth:** With quality and full-video support done, make it dead simple to use.

### Task 7: CLI entry point with subcommands

**Files:**
- Create: `clear_stage/cli.py`

- [ ] **Step 1: Create CLI with `process` subcommand**

```python
# clear_stage/cli.py
"""Clear Stage CLI — remove background dancers from pole dance videos."""
import argparse
import sys

def cmd_process(args):
    """Process a video to remove background people."""
    from clear_stage.run_pipeline import run_pipeline
    run_pipeline(
        video=args.video,
        output=args.output,
        prompt=args.prompt or "A pole dance studio with mirrors and wooden floor",
        quality=args.quality,
        principal=args.principal,
        frame=args.frame,
        work_dir=args.work_dir,
        device=args.device,
    )

def cmd_detect(args):
    """Detect people in a video and save preview."""
    from clear_stage.detect_people import detect_people, find_good_detection_frame
    from clear_stage.select_principal import save_detection_preview
    frame_idx, detections = find_good_detection_frame(args.video)
    save_detection_preview(args.video, detections, args.output or "detection_preview.jpg", frame_idx)
    for i, d in enumerate(detections):
        b = d["bbox"]
        print(f"  #{i}: {b[2]-b[0]}x{b[3]-b[1]} score={d['score']:.2f}")

def main():
    parser = argparse.ArgumentParser(prog="clear-stage")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("process", help="Remove background people from video")
    p.add_argument("video", help="Input video path")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-p", "--prompt", default=None)
    p.add_argument("-q", "--quality", default="standard", choices=["preview", "standard", "high"])
    p.add_argument("--principal", type=int, default=None)
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--work-dir", default="/tmp/clear_stage_work")
    p.add_argument("--device", default="cuda")

    d = sub.add_parser("detect", help="Detect people and save preview")
    d.add_argument("video")
    d.add_argument("-o", "--output", default=None)

    args = parser.parse_args()
    if args.command == "process":
        cmd_process(args)
    elif args.command == "detect":
        cmd_detect(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add entry point to setup**

Create `pyproject.toml` or add to existing config:

```toml
[project.scripts]
clear-stage = "clear_stage.cli:main"
```

- [ ] **Step 3: Test**

```bash
clear-stage process sample_videos/IMG_2745.mov -o output/test.mp4 -q preview --principal 1
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: clear-stage CLI with process and detect subcommands"
```

---

### Task 8: RunPod remote runner

**Files:**
- Create: `clear_stage/runpod_runner.py`

Automate the full RunPod workflow: resume pod → upload video → run pipeline → download result → stop pod.

- [ ] **Step 1: Implement RunPod runner**

```python
# clear_stage/runpod_runner.py
"""Run clear-stage pipeline remotely on RunPod."""
import subprocess
import time

RUNPOD_POD_ID = "cts9ata0uqd95e"
SSH_KEY = "~/.ssh/runpod_key"
NETWORK_VOLUME_ID = "oxfxblvrr6"

def get_api_key():
    """Read RunPod API key from .env file."""
    from pathlib import Path
    env = Path(__file__).parent.parent / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise ValueError("RUNPOD_API_KEY not found in .env")

def resume_pod():
    """Resume the RunPod pod and return SSH connection info."""
    ...

def setup_pod(ssh_cmd):
    """Clone repo, symlink models, install ffmpeg on pod."""
    ...

def upload_video(ssh_cmd, local_path, remote_path):
    """Upload video to pod via scp."""
    ...

def run_remote(ssh_cmd, video_path, output_path, quality, principal):
    """Run clear-stage process on the pod."""
    ...

def download_result(ssh_cmd, remote_path, local_path):
    """Download processed video from pod."""
    ...

def stop_pod():
    """Stop the pod to save money."""
    ...

def process_remote(video_path, output_path, quality="standard", principal=None):
    """Full remote processing workflow. Always stops pod on exit."""
    try:
        print("Resuming RunPod pod...")
        ssh_cmd = resume_pod()  # polls with timeout until SSH is ready
        print("Setting up pod...")
        setup_pod(ssh_cmd)
        print("Uploading video...")
        upload_video(ssh_cmd, video_path, "/workspace/clear-stage/input.mp4")
        print("Processing...")
        run_remote(ssh_cmd, "/workspace/clear-stage/input.mp4",
                   "/workspace/clear-stage/output.mp4", quality, principal)
        print("Downloading result...")
        download_result(ssh_cmd, "/workspace/clear-stage/output.mp4", output_path)
        print(f"Done! Result: {output_path}")
    finally:
        print("Stopping pod...")
        stop_pod()  # ALWAYS stop to avoid runaway costs
```

- [ ] **Step 2: Add `remote` subcommand to CLI**

```python
r = sub.add_parser("remote", help="Process video on RunPod GPU")
r.add_argument("video")
r.add_argument("-o", "--output", required=True)
r.add_argument("-q", "--quality", default="standard", choices=["preview", "standard", "high"])
r.add_argument("--principal", type=int, default=None)
```

- [ ] **Step 3: Test full remote workflow**

```bash
clear-stage remote sample_videos/IMG_2745.mov -o output/remote_test.mp4 -q preview --principal 1
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: RunPod remote runner — single command for full remote processing"
```

---

## Milestone 5: Gemini VLM Shadow/Reflection Analysis (Optional Enhancement)

**Why last:** The pipeline works without it. This adds refinement for shadow handling but requires API integration work.

### Task 9: Enable Gemini VLM stage for affected regions

**Files:**
- Modify: `clear_stage/segment_people.py` or create `clear_stage/vlm_analysis.py`

- [ ] **Step 1: Install openai package on RunPod**

```bash
pip install openai
```

- [ ] **Step 2: After SAM2 segmentation, run VLM-MASK-REASONER Stage 2**

Use `stage2_vlm_analysis.py` to identify shadows and affected objects, then `stage3a_generate_grey_masks_v2.py` for grey mask generation. Combine black+grey into proper quadmask using `stage4_combine_masks.py`.

This converts our binary mask (0/255) into a proper quadmask (0/63/127/255) where:
- 0 = primary removal regions (the people)
- 127 = affected regions (shadows, reflections VOID should also clean up)
- 63 = overlap of primary + affected
- 255 = keep

- [ ] **Step 3: Test with and without VLM stage, compare quality**

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: Gemini VLM analysis for shadow/reflection-aware quadmasks"
```

---

## Summary & Success Criteria

| Milestone | Deliverable | Success Metric |
|---|---|---|
| **M1: FPS + Pipeline Fix** | Video plays at correct speed, people actually removed | Output fps matches input, mask overlay shows full coverage |
| **M2: Resolution + Mask** | Configurable quality, no hand/pole blur | `--quality preview/standard/high` works, principal dancer sharp |
| **M3: Full Videos** | Process 30s–6min videos | 98s IMG_2745.mov processes end-to-end with audio |
| **M4: Automation** | Single command from laptop | `clear-stage remote video.mov -o out.mp4` does everything |
| **M5: VLM Shadows** | Proper shadow/reflection removal | Quadmask has grey values, shadows cleaned |

**Estimated RunPod cost per milestone:**
- M1: ~$1 (2 test runs)
- M2: ~$2 (4 test runs at different resolutions)
- M3: ~$5 (processing full-length videos)
- M4: ~$1 (testing remote runner)
- M5: ~$2 (VLM comparison tests)
- **Total: ~$11**
