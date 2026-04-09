def test_preview_resolution_portrait():
    from clear_stage.resolution import get_sample_size
    s = get_sample_size(720, 1280, quality="preview")
    h, w = map(int, s.split("x"))
    assert h > w  # portrait
    assert h * w < 100000
    assert h % 16 == 0 and w % 16 == 0


def test_standard_resolution_portrait():
    from clear_stage.resolution import get_sample_size
    s = get_sample_size(720, 1280, quality="standard")
    h, w = map(int, s.split("x"))
    assert h > w
    assert 200000 < h * w < 300000


def test_high_resolution_portrait():
    from clear_stage.resolution import get_sample_size
    s = get_sample_size(720, 1280, quality="high")
    h, w = map(int, s.split("x"))
    assert h > w
    assert h * w > 400000


def test_standard_resolution_landscape():
    from clear_stage.resolution import get_sample_size
    s = get_sample_size(1280, 720, quality="standard")
    h, w = map(int, s.split("x"))
    assert w > h  # landscape
    assert 200000 < h * w < 300000


def test_resolution_preserves_aspect_ratio():
    from clear_stage.resolution import get_sample_size
    for q in ["preview", "standard", "high"]:
        s = get_sample_size(720, 1280, quality=q)
        h, w = map(int, s.split("x"))
        aspect = w / h
        expected = 720 / 1280
        assert abs(aspect - expected) < 0.1


def test_square_video():
    from clear_stage.resolution import get_sample_size
    s = get_sample_size(1080, 1080, quality="standard")
    h, w = map(int, s.split("x"))
    assert abs(h - w) <= 16  # roughly square
