# Pole Dance Studio Background Person Removal

**Date:** 2026-04-08
**Status:** Draft

## Problem

When recording pole dancing in a studio, other dancers are visible in the background. The user wants a tool that removes all background people from the video while keeping the principal dancer (themselves) and their reflections in studio mirrors.

## Solution

A pipeline that combines SAM 3.1 (person segmentation & tracking) with Netflix's VOID model (interaction-aware video inpainting) to automatically detect, mask, and remove background dancers — including their reflections and shadows.

## Architecture

```
Upload Video (mobile web)
    |
    v
[Person Detection + Selection]
  - Use SAM 3.1 text prompting OR GroundingDINO to detect all people in frame 1
  - Show detected people in web UI with colored overlays
  - User taps/clicks the person to KEEP
  - SAM 3.1 (or SAM 2 fallback) tracks all people across all frames
  - NOTE: SAM 3.1 text-prompt video tracking needs verification.
    Fallback: use GroundingDINO for detection + SAM 2 point-prompt tracking
    (the proven approach from the existing pipeline)
    |
    v
[VLM-MASK-REASONER pipeline (Stages 2-4)]
  - Stage 2: Gemini VLM analyzes reflections, shadows, interactions
  - Stage 3: Generates grey masks for affected regions (mirror reflections, shadows)
  - Stage 4: Combines into quadmask format (0=remove, 63=overlap, 127=affected, 255=keep)
    |
    v
[Chunking]
  - Downsample from 30fps to 12fps (reduces chunks ~2.5x)
  - Split into 197-frame chunks with ~20 frame overlap
    (VOID ingests up to 197 frames per call; the 85-frame temporal_window_size
    is an internal multidiffusion window, not the chunk boundary)
  - Each chunk paired with its mask slice and background prompt
  - Strip audio track before processing; re-mux after stitching
    |
    v
[Parallel VOID Pass 1 — RunPod Serverless]
  - Fan out chunks to N parallel A100 GPU workers
  - Each runs VOID Pass 1 (50 inference steps)
  - ~2-5 min per chunk per worker
    |
    v
[Stitch & Upsample]
  - Linear crossfade blending in overlap regions
  - Frame interpolation (RIFE) from 12fps back to 30fps
  - Output final video
```

## User Workflow (v1)

1. Open mobile-friendly web page
2. Upload pole dance video (30s - 6min)
3. See first frame with all detected people highlighted in different colors
4. Tap/click yourself to mark as "keep"
5. Optionally enter a background description (e.g., "A pole dance studio with mirrors and wooden floor") or let Gemini auto-generate later
6. Wait for processing (5-30 min depending on length and parallelism)
7. Get notification, download result

## Pricing Model

- **Preview tier**: Low resolution (192x336), fewer inference steps (~15), fast/cheap. Lets user verify the removal looks correct before paying.
- **Full quality tier**: 384x672 with 50 steps. Higher resolution options as a future tier.

## Key Technical Decisions

### Person Detection & Segmentation

**Preferred: SAM 3.1** (released March 2026):
- Native text prompting ("person") eliminates the need for a separate detector
- Object Multiplex tracks up to 16 objects in a single forward pass (7x faster)
- Backward compatible with SAM 2's visual prompting

**Fallback: GroundingDINO + SAM 2** (proven approach):
- SAM 3.1's text-prompt video tracking is new and needs verification
- The existing VLM-MASK-REASONER Stage 1 uses SAM 2 with point prompts from a GUI
- GroundingDINO can provide bounding box centers as automatic point prompts for SAM 2
- This is lower risk but requires an extra model dependency

**Implementation plan:** Start with GroundingDINO + SAM 2 (known to work), swap to SAM 3.1 once verified.

### VOID over EraserDiT or ProPainter

VOID is the right choice despite being slower because:
- Interaction-aware removal handles mirror reflections and shadows — critical for pole studios
- 64.8% human preference over Runway in blind tests
- Open source (Apache 2.0)
- EraserDiT is 10-20x faster but lacks physics/interaction awareness

### Parallel chunk processing on RunPod

- Chunks are independent after mask generation — embarrassingly parallel
- RunPod Community Cloud A100 at $0.89/hr is cost-effective for testing
- RunPod Serverless can burst to many workers for faster turnaround

### FPS downsampling

- Process at 12fps (VOID's native rate) instead of 30fps
- Reduces chunk count by ~2.5x
- Use RIFE frame interpolation to restore 30fps output
- Acceptable quality trade-off for testing; revisit if motion artifacts appear

## Compute & Cost Estimates

VOID ingests up to 197 frames per call (with internal 85-frame multidiffusion windows). At 12fps:

| Video length | Frames @12fps | Chunks (197f, 20f overlap) | Serial (1 GPU) | 4 GPUs parallel | Est. cost |
|---|---|---|---|---|---|
| 30 seconds | ~360 | 2 | ~5-10 min | ~5-10 min | ~$0.30 |
| 2 minutes | ~1440 | 8 | ~20-40 min | ~5-10 min | ~$1.50 |
| 6 minutes | ~4320 | 25 | ~60-125 min | ~15-30 min | ~$5 |

## Known Limitations (v1)

- **People entering/exiting mid-video:** Detection runs on frame 1 only.
- **Heavy dancer overlap:** Inpainting quality may degrade when dancers cross.
- **Chunk boundary artifacts:** Linear crossfade may show subtle flickering.
- **Mirror edge cases:** Overlapping reflections may confuse the VLM.
- **Resolution:** 384x672 output for v1. Input downscaled on ingest.
- **RIFE interpolation artifacts:** Fast pole spins may produce motion artifacts at 12fps->30fps.
- **Audio:** Stripped before processing, re-muxed after.
- **Tracking drift:** May need periodic re-initialization on keyframes for long videos.

## Dependencies

| Component | Version | Purpose |
|---|---|---|
| SAM 3.1 | Latest | Person segmentation & tracking |
| VOID (Pass 1) | netflix/void-model | Video inpainting |
| CogVideoX-Fun-V1.5-5b-InP | alibaba-pai | Base model (VOID fine-tuned on this) |
| GroundingDINO | Latest | Person detection (fallback) |
| Gemini API | Latest | VLM scene analysis |
| RIFE | Latest | Frame interpolation |
| RunPod Serverless | - | GPU compute |

## Success Criteria

- Background dancers cleanly removed from a 30-second test clip
- Principal dancer's mirror reflection preserved; background dancers' reflections removed
- No obvious chunk boundary artifacts
- Natural-looking pole and studio background
- Processing under 10 minutes for a 30-second clip
- Audio preserved in output
