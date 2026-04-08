"""
End-to-end pipeline: detect → select → mask → chunk → VOID → stitch.

Usage:
    python -m clear_stage.run_pipeline \
        --video sample_videos/IMG_2745.mov \
        --output output/result.mp4 \
        --prompt "A pole dance studio with mirrors and wooden floor" \
        --work-dir /tmp/clear_stage_work
"""
import argparse
import subprocess
from pathlib import Path

from clear_stage.detect_people import detect_people, find_good_detection_frame
from clear_stage.select_principal import select_principal_cli, save_detection_preview
from clear_stage.generate_points_config import generate_config, save_config
from clear_stage.chunk_video import prepare_chunks
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
        cwd=str(void_root),
        check=True,
    )


def compute_void_sample_size(width: int, height: int) -> str:
    """Compute the VOID sample_size string (HxW) preserving aspect ratio.

    VOID's default is 384x672 (landscape). Dimensions must be divisible by 16
    for the VAE. We pick the closest fit within the ~250k pixel budget
    (384*672 = 258048) that preserves the input aspect ratio.
    """
    target_pixels = 384 * 672  # ~258k pixels, VOID's default budget
    aspect = width / height

    # Solve: h * w = target_pixels, w/h = aspect
    # h = sqrt(target_pixels / aspect), w = h * aspect
    h = int((target_pixels / aspect) ** 0.5)
    w = int(h * aspect)

    # Round to nearest multiple of 16 (VAE requirement)
    h = max(16, (h // 16) * 16)
    w = max(16, (w // 16) * 16)

    return f"{h}x{w}"


def run_void_inference(
    data_rootdir: str, seq_names: list[str],
    sample_size: str = "384x672",
    transformer_path: str = "void-model/void_pass1.safetensors",
    save_path: str | None = None,
) -> None:
    """Run VOID Pass 1 inference.

    predict_v2v.py uses ml_collections config flags (Python .py files).
    sample_size: 'HxW' string matching the input video aspect ratio.
    """
    repo_root = Path(__file__).parent.parent
    script = repo_root / "void-model" / "inference" / "cogvideox_fun" / "predict_v2v.py"
    py_config = str(repo_root / "void-model" / "config" / "quadmask_cogvideox.py")

    if save_path is None:
        save_path = str(Path(data_rootdir) / "void_output")

    cmd = [
        "python", str(script),
        f"--config={py_config}",
        f"--config.data.data_rootdir={data_rootdir}",
        f"--config.data.sample_size={sample_size}",
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
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame to use for detection (default: auto-find)")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Step 1: Detect people
    print("\n=== Step 1: Detecting people ===")
    if args.frame is not None:
        frame_idx = args.frame
        detections = detect_people(args.video, frame_idx)
    else:
        print("Auto-finding best frame for detection...")
        frame_idx, detections = find_good_detection_frame(args.video)
        print(f"Using frame {frame_idx}")

    if not detections:
        print("No people detected. Nothing to remove.")
        subprocess.run(["cp", args.video, args.output], check=True)
        return
    print(f"Found {len(detections)} people")

    # Step 2: Select principal
    preview = str(work / "detection_preview.jpg")
    save_detection_preview(args.video, detections, preview, frame_idx)
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
    config = generate_config(
        args.video, mask_dir, detections, principal_idx, frame_idx,
    )
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

    # Step 5: VOID inference (serial — parallelize on RunPod later)
    print("\n=== Step 5: VOID inference ===")
    # Detect video dimensions and compute aspect-ratio-preserving sample size
    from clear_stage.chunk_video import get_video_info
    video_info = get_video_info(args.video)
    sample_size = compute_void_sample_size(video_info["width"], video_info["height"])
    print(f"Input: {video_info['width']}x{video_info['height']} -> VOID sample_size: {sample_size}")
    run_void_inference(
        chunks_dir, [c["chunk_name"] for c in chunk_infos],
        sample_size=sample_size,
        transformer_path=args.transformer_path,
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
