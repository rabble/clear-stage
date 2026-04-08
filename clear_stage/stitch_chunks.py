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


def find_void_output(chunk_dir: str, chunk_name: str) -> str:
    """Find VOID output file by glob pattern (output naming varies)."""
    import glob
    patterns = [
        f"{chunk_dir}/{chunk_name}/**/output*.mp4",
        f"{chunk_dir}/{chunk_name}/**/*-fg=*.mp4",
        f"{chunk_dir}/**/{chunk_name}*.mp4",
        f"{chunk_dir}/void_output/**/{chunk_name}*.mp4",
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No VOID output found for chunk {chunk_name} in {chunk_dir}")


def stitch_chunks(
    chunk_infos: list[dict], chunk_output_dir: str, output_path: str,
    overlap: int = 20, fps: float = 12.0,
) -> str:
    """Stitch processed chunks with linear crossfade in overlap regions."""
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
