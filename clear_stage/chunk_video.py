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
    """Calculate number of frames after downsampling."""
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
    """Downsample video to target FPS."""
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-r", str(target_fps),
                    "-c:v", "libx264", "-crf", "18", "-an", output_path],
                   check=True, capture_output=True)
    return output_path


def split_video_chunk(video_path: str, start: int, end: int, fps: float, output_path: str) -> str:
    """Extract a chunk of video by frame range."""
    start_t, dur = start / fps, (end - start) / fps
    subprocess.run(["ffmpeg", "-y", "-ss", str(start_t), "-i", video_path,
                    "-t", str(dur), "-c:v", "libx264", "-crf", "18", "-an", output_path],
                   check=True, capture_output=True)
    return output_path


def prepare_chunks(
    video_path: str, mask_path: str, prompt: str, output_dir: str,
    target_fps: float | None = 24.0, max_frames: int = 197, overlap: int = 20,
) -> list[dict]:
    """Split video+mask into chunks for VOID processing.

    If target_fps is set, downsamples first (cheaper, more temporal context per chunk).
    If target_fps is None, uses source fps (preserves quality, costs more chunks).

    Returns list of chunk info dicts including 'source_fps' for correct output encoding.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_info = get_video_info(video_path)
    source_fps = source_info["fps"]

    # Extract audio for later remux
    extract_audio(video_path, str(out / "audio.aac"))

    # Optionally downsample
    if target_fps is not None and target_fps < source_fps:
        work_video = str(out / "downsampled_video.mp4")
        work_mask = str(out / "downsampled_mask.mp4")
        downsample_video(video_path, work_video, target_fps)
        downsample_video(mask_path, work_mask, target_fps)
        process_fps = target_fps
    else:
        work_video = video_path
        work_mask = mask_path
        process_fps = source_fps

    work_info = get_video_info(work_video)
    chunks = calculate_chunks(work_info["total_frames"], max_frames, overlap)

    chunk_infos = []
    for i, (start, end) in enumerate(chunks):
        cd = out / f"chunk_{i:03d}"
        cd.mkdir(exist_ok=True)
        split_video_chunk(work_video, start, end, process_fps, str(cd / "input_video.mp4"))
        split_video_chunk(work_mask, start, end, process_fps, str(cd / "quadmask_0.mp4"))
        with open(cd / "prompt.json", "w") as f:
            json.dump({"bg": prompt}, f)
        chunk_infos.append({"chunk_dir": str(cd), "chunk_name": f"chunk_{i:03d}",
                           "start": start, "end": end,
                           "source_fps": source_fps, "process_fps": process_fps})
    return chunk_infos
