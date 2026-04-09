"""
End-to-end pipeline: detect → segment → VOID inpaint → output.

Usage:
    python -m clear_stage.run_pipeline \
        --video sample_videos/IMG_2745.mov \
        --output output/result.mp4 \
        --prompt "A pole dance studio with mirrors and wooden floor" \
        --principal 1
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from clear_stage.detect_people import detect_people, find_good_detection_frame
from clear_stage.select_principal import select_principal_cli, save_detection_preview
from clear_stage.segment_people import segment_background_people, save_mask_video, save_mask_overlay
from clear_stage.chunk_video import get_video_info, extract_audio
from clear_stage.stitch_chunks import remux_audio
from clear_stage.resolution import get_sample_size


def run_void_inference(
    data_rootdir: str, seq_names: list[str],
    sample_size: str = "384x672",
    transformer_path: str = "void-model/void_pass1.safetensors",
    save_path: str | None = None,
) -> None:
    """Run VOID Pass 1 inference."""
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


def run_pipeline(
    video: str,
    output: str,
    prompt: str = "A pole dance studio with mirrors and wooden floor",
    quality: str = "standard",
    principal: int | None = None,
    frame: int | None = None,
    work_dir: str = "/tmp/clear_stage_work",
    device: str = "cuda",
    transformer_path: str = "void-model/void_pass1.safetensors",
    sam2_checkpoint: str | None = None,
) -> str:
    """Run the full clear-stage pipeline.

    Returns path to the output video.
    """
    repo_root = Path(__file__).parent.parent
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    if sam2_checkpoint is None:
        sam2_checkpoint = str(repo_root / "void-model" / "sam2_hiera_large.pt")

    # Step 1: Detect people
    print("\n=== Step 1: Detecting people ===")
    if frame is not None:
        frame_idx = frame
        detections = detect_people(video, frame_idx)
    else:
        print("Auto-finding best frame for detection...")
        frame_idx, detections = find_good_detection_frame(video)
        print(f"Using frame {frame_idx}")

    if not detections:
        print("No people detected. Nothing to remove.")
        shutil.copy(video, output)
        return output
    print(f"Found {len(detections)} people")

    # Step 2: Select principal dancer
    preview = str(work / "detection_preview.jpg")
    save_detection_preview(video, detections, preview, frame_idx)
    print(f"Preview saved: {preview}")

    if principal is not None:
        principal_idx = principal
    else:
        principal_idx = select_principal_cli(detections)

    if len(detections) == 1:
        print("Only one person detected (the principal). Nothing to remove.")
        shutil.copy(video, output)
        return output

    # Step 3: Segment background people with SAM2
    print("\n=== Step 3: Segmenting background people ===")
    mask_video = segment_background_people(
        video, detections, principal_idx,
        sam2_checkpoint=sam2_checkpoint,
        frame_idx=frame_idx, device=device,
    )

    # Save mask overlay for debugging
    save_mask_overlay(video, mask_video, str(work / "mask_overlay.jpg"), frame_idx)
    print(f"Mask overlay saved: {work / 'mask_overlay.jpg'}")

    # Step 4: Set up VOID input structure
    print("\n=== Step 4: Preparing VOID input ===")
    video_info = get_video_info(video)
    sample_size = get_sample_size(video_info["width"], video_info["height"], quality)
    print(f"Input: {video_info['width']}x{video_info['height']} "
          f"-> VOID sample_size: {sample_size} (quality={quality})")

    void_dir = str(work / "void_input")
    seq_dir = os.path.join(void_dir, "seq")
    os.makedirs(seq_dir, exist_ok=True)

    shutil.copy(video, os.path.join(seq_dir, "input_video.mp4"))
    save_mask_video(mask_video, os.path.join(seq_dir, "quadmask_0.mp4"),
                    fps=video_info["fps"])
    with open(os.path.join(seq_dir, "prompt.json"), "w") as f:
        json.dump({"bg": prompt}, f)

    # Step 5: Run VOID inference
    print("\n=== Step 5: VOID inference ===")
    void_output_dir = str(work / "void_output")
    run_void_inference(
        void_dir, ["seq"],
        sample_size=sample_size,
        transformer_path=transformer_path,
        save_path=void_output_dir,
    )

    # Step 6: Find output and remux audio
    print("\n=== Step 6: Finalizing output ===")
    import glob
    void_outputs = glob.glob(f"{void_output_dir}/**/*.mp4", recursive=True)
    # Filter out _tuple files
    void_outputs = [f for f in void_outputs if "_tuple" not in f]
    if not void_outputs:
        raise FileNotFoundError(f"No VOID output found in {void_output_dir}")

    void_result = void_outputs[0]
    print(f"VOID output: {void_result}")

    # Extract and remux audio from original
    audio_path = str(work / "audio.aac")
    has_audio = extract_audio(video, audio_path)
    if has_audio:
        remux_audio(void_result, audio_path, output)
        print(f"Audio remuxed into output")
    else:
        shutil.copy(void_result, output)

    print(f"\nDone! Output: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Remove background dancers from pole dance video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="A pole dance studio with mirrors and wooden floor",
                        help="Background description after removal")
    parser.add_argument("--work-dir", default="/tmp/clear_stage_work")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transformer-path", default="void-model/void_pass1.safetensors")
    parser.add_argument("--principal", type=int, default=None,
                        help="Skip interactive selection, use this index")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame to use for detection (default: auto-find)")
    parser.add_argument("--quality", default="standard",
                        choices=["preview", "standard", "high"],
                        help="Output resolution quality")
    args = parser.parse_args()

    run_pipeline(
        video=args.video,
        output=args.output,
        prompt=args.prompt,
        quality=args.quality,
        principal=args.principal,
        frame=args.frame,
        work_dir=args.work_dir,
        device=args.device,
        transformer_path=args.transformer_path,
    )


if __name__ == "__main__":
    main()
