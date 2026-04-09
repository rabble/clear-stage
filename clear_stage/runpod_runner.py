"""Run clear-stage pipeline remotely on RunPod.

Automates: resume pod → setup → upload video → process → download result → stop pod.
"""
import os
import subprocess
import time
from pathlib import Path

# Pod config — update these if you create a new pod
RUNPOD_POD_ID = "cts9ata0uqd95e"
SSH_KEY = os.path.expanduser("~/.ssh/runpod_key")


def _load_env():
    """Load .env file from project root."""
    env_path = Path(__file__).parent.parent / ".env"
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _runpod_api(query: str) -> dict:
    """Execute a RunPod GraphQL query."""
    import json
    env = _load_env()
    api_key = env.get("RUNPOD_API_KEY")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY not found in .env")

    result = subprocess.run(
        ["curl", "-s",
         f"https://api.runpod.io/graphql?api_key={api_key}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"query": query})],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def resume_pod(timeout: int = 120) -> str:
    """Resume pod and wait for SSH. Returns SSH command prefix."""
    import json

    env = _load_env()
    api_key = env.get("RUNPOD_API_KEY")

    # Resume via SDK
    subprocess.run(
        ["python3", "-c", f"""
import runpod
runpod.api_key = '{api_key}'
runpod.resume_pod('{RUNPOD_POD_ID}', gpu_count=1)
"""],
        check=True, capture_output=True,
    )

    # Poll for SSH readiness
    start = time.time()
    while time.time() - start < timeout:
        try:
            data = _runpod_api(
                f'{{ pod(input: {{podId: "{RUNPOD_POD_ID}"}}) '
                f'{{ runtime {{ ports {{ ip isIpPublic privatePort publicPort }} }} }} }}'
            )
            pod = data.get("data", {}).get("pod", {})
            rt = pod.get("runtime") or {}
            ports = rt.get("ports") or []
            ssh = [p for p in ports if p["privatePort"] == 22 and p.get("isIpPublic")]
            if ssh:
                s = ssh[0]
                ssh_cmd = f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no root@{s['ip']} -p {s['publicPort']}"
                # Test connection
                r = subprocess.run(
                    ssh_cmd.split() + ["echo", "ready"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    print(f"Pod ready: {s['ip']}:{s['publicPort']}")
                    return ssh_cmd
        except Exception:
            pass
        time.sleep(10)

    raise TimeoutError(f"Pod did not become ready within {timeout}s")


def setup_pod(ssh_cmd: str) -> None:
    """Clone repo, symlink models, install ffmpeg on pod."""
    setup_script = """
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
cd /workspace
if [ ! -d clear-stage/.git ]; then
    git clone --recursive https://github.com/rabble/clear-stage.git
else
    cd clear-stage && git pull
    cd /workspace
fi
cd clear-stage
mkdir -p sample_videos output
ln -sf /runpod-volume/clear-stage-venv .venv
ln -sf /runpod-volume/models/void_pass1.safetensors void-model/void_pass1.safetensors
ln -sf /runpod-volume/models/sam2_hiera_large.pt void-model/sam2_hiera_large.pt
ln -sf /runpod-volume/models/groundingdino_swint_ogc.pth groundingdino_swint_ogc.pth
ln -sfn /runpod-volume/models/CogVideoX-Fun-V1.5-5b-InP void-model/CogVideoX-Fun-V1.5-5b-InP
ln -sfn /runpod-volume/models/CogVideoX-Fun-V1.5-5b-InP ./CogVideoX-Fun-V1.5-5b-InP
echo "Pod setup complete"
"""
    subprocess.run(ssh_cmd.split() + [setup_script], check=True)


def upload_video(ssh_cmd: str, local_path: str, remote_path: str) -> None:
    """Upload video to pod via scp."""
    parts = ssh_cmd.split()
    # Extract host and port from ssh command
    key = parts[parts.index("-i") + 1]
    host = parts[-1].split("@")[-1] if "@" in parts[-1] else parts[-1]
    port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
    user_host = [p for p in parts if "@" in p][0]

    subprocess.run(
        ["scp", "-i", key, "-P", port, "-o", "StrictHostKeyChecking=no",
         local_path, f"{user_host}:{remote_path}"],
        check=True,
    )


def download_result(ssh_cmd: str, remote_path: str, local_path: str) -> None:
    """Download processed video from pod."""
    parts = ssh_cmd.split()
    key = parts[parts.index("-i") + 1]
    port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
    user_host = [p for p in parts if "@" in p][0]

    subprocess.run(
        ["scp", "-i", key, "-P", port, "-o", "StrictHostKeyChecking=no",
         f"{user_host}:{remote_path}", local_path],
        check=True,
    )


def run_remote_pipeline(ssh_cmd: str, remote_video: str, remote_output: str,
                        quality: str, principal: int | None, prompt: str) -> None:
    """Run the pipeline on the pod."""
    cmd = (
        f"cd /workspace/clear-stage && source .venv/bin/activate && "
        f"export GEMINI_API_KEY=$(grep GEMINI_API_KEY .env 2>/dev/null | cut -d= -f2) && "
        f"python -m clear_stage.run_pipeline "
        f"--video {remote_video} --output {remote_output} "
        f"--quality {quality} "
        f'--prompt "{prompt}"'
    )
    if principal is not None:
        cmd += f" --principal {principal}"

    subprocess.run(ssh_cmd.split() + [cmd], check=True)


def stop_pod() -> None:
    """Stop the pod to save money."""
    env = _load_env()
    api_key = env.get("RUNPOD_API_KEY")
    subprocess.run(
        ["python3", "-c", f"""
import runpod
runpod.api_key = '{api_key}'
runpod.stop_pod('{RUNPOD_POD_ID}')
print('Pod stopped')
"""],
        capture_output=True,
    )


def process_remote(
    video_path: str,
    output_path: str,
    quality: str = "standard",
    principal: int | None = None,
    prompt: str = "A pole dance studio with mirrors and wooden floor",
) -> str:
    """Full remote processing workflow. Always stops pod on exit.

    Returns path to the output video.
    """
    remote_video = "/workspace/clear-stage/input_video.mp4"
    remote_output = "/workspace/clear-stage/output_video.mp4"

    try:
        print("Resuming RunPod pod...")
        ssh_cmd = resume_pod()

        print("Setting up pod...")
        setup_pod(ssh_cmd)

        print(f"Uploading {video_path}...")
        upload_video(ssh_cmd, video_path, remote_video)

        print(f"Processing (quality={quality})...")
        run_remote_pipeline(ssh_cmd, remote_video, remote_output,
                           quality, principal, prompt)

        print(f"Downloading result...")
        download_result(ssh_cmd, remote_output, output_path)

        print(f"Done! Result: {output_path}")
        return output_path

    finally:
        print("Stopping pod...")
        stop_pod()
