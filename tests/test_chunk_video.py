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


def test_calculate_chunks_long_video():
    """6 min video at 12fps = 4320 frames."""
    from clear_stage.chunk_video import calculate_chunks
    chunks = calculate_chunks(total_frames=4320, max_frames=197, overlap=20)
    # Verify full coverage
    assert chunks[0][0] == 0
    assert chunks[-1][1] >= 4320
    # Verify consistent overlap
    for i in range(len(chunks) - 1):
        assert chunks[i][1] - chunks[i + 1][0] == 20


def test_downsample_frames():
    from clear_stage.chunk_video import calculate_downsampled_frames
    assert calculate_downsampled_frames(900, 30, 12) == 360
    assert calculate_downsampled_frames(3600, 60, 12) == 720
