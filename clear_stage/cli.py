"""Clear Stage CLI — remove background dancers from pole dance videos."""
import argparse
import sys


def cmd_process(args):
    """Process a video locally (requires GPU)."""
    from clear_stage.run_pipeline import run_pipeline
    run_pipeline(
        video=args.video,
        output=args.output,
        prompt=args.prompt,
        quality=args.quality,
        principal=args.principal,
        frame=args.frame,
        work_dir=args.work_dir,
        device=args.device,
    )


def cmd_detect(args):
    """Detect people in a video and save preview image."""
    from clear_stage.detect_people import detect_people, find_good_detection_frame
    from clear_stage.select_principal import save_detection_preview

    frame_idx, detections = find_good_detection_frame(args.video)
    output = args.output or "detection_preview.jpg"
    save_detection_preview(args.video, detections, output, frame_idx)
    print(f"Frame {frame_idx}: {len(detections)} people detected")
    for i, d in enumerate(detections):
        b = d["bbox"]
        print(f"  #{i}: {b[2]-b[0]}x{b[3]-b[1]} score={d['score']:.2f}")
    print(f"Preview: {output}")


def cmd_remote(args):
    """Process a video on RunPod GPU."""
    from clear_stage.runpod_runner import process_remote
    process_remote(
        video_path=args.video,
        output_path=args.output,
        quality=args.quality,
        principal=args.principal,
        prompt=args.prompt,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="clear-stage",
        description="Remove background people from pole dance studio videos",
    )
    sub = parser.add_subparsers(dest="command")

    # process (local GPU)
    p = sub.add_parser("process", help="Process video locally (requires GPU)")
    p.add_argument("video", help="Input video path")
    p.add_argument("-o", "--output", required=True, help="Output video path")
    p.add_argument("-p", "--prompt", default="A pole dance studio with mirrors and wooden floor")
    p.add_argument("-q", "--quality", default="standard", choices=["preview", "standard", "high"])
    p.add_argument("--principal", type=int, default=None, help="Person index to keep (skip selection)")
    p.add_argument("--frame", type=int, default=None, help="Frame for detection (default: auto)")
    p.add_argument("--work-dir", default="/tmp/clear_stage_work")
    p.add_argument("--device", default="cuda")

    # detect (preview only)
    d = sub.add_parser("detect", help="Detect people and save preview image")
    d.add_argument("video", help="Input video path")
    d.add_argument("-o", "--output", default=None, help="Preview image path")

    # remote (RunPod)
    r = sub.add_parser("remote", help="Process video on RunPod GPU")
    r.add_argument("video", help="Input video path (local)")
    r.add_argument("-o", "--output", required=True, help="Output video path (local)")
    r.add_argument("-p", "--prompt", default="A pole dance studio with mirrors and wooden floor")
    r.add_argument("-q", "--quality", default="standard", choices=["preview", "standard", "high"])
    r.add_argument("--principal", type=int, default=None)

    args = parser.parse_args()
    if args.command == "process":
        cmd_process(args)
    elif args.command == "detect":
        cmd_detect(args)
    elif args.command == "remote":
        cmd_remote(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
