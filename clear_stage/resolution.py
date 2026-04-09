"""Resolution presets for VOID processing.

VOID's sample_size format is 'HxW' (height x width).
Dimensions must be divisible by 16 for the VAE.
"""

PRESETS = {
    "preview":  64512,   # ~254x254 equivalent, fast testing
    "standard": 258048,  # 384x672, VOID's default
    "high":     516096,  # ~543x950, 2x pixels
}


def get_sample_size(width: int, height: int, quality: str = "standard") -> str:
    """Compute VOID sample_size (HxW) preserving aspect ratio at given quality.

    Args:
        width: Input video width in pixels
        height: Input video height in pixels
        quality: One of 'preview', 'standard', 'high'

    Returns:
        String like '672x368' for portrait, '368x672' for landscape
    """
    target_pixels = PRESETS[quality]
    aspect = width / height

    h = int((target_pixels / aspect) ** 0.5)
    w = int(h * aspect)

    h = max(16, (h // 16) * 16)
    w = max(16, (w // 16) * 16)

    return f"{h}x{w}"
