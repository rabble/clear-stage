def test_generate_config_excludes_principal():
    """Config should contain points for all people EXCEPT the principal."""
    from clear_stage.generate_points_config import generate_config

    detections = [
        {"bbox": [100, 100, 200, 400], "score": 0.9},
        {"bbox": [400, 100, 500, 400], "score": 0.8},  # principal
        {"bbox": [700, 100, 800, 400], "score": 0.7},
    ]

    config = generate_config(
        video_path="test.mp4", output_dir="out", detections=detections, principal_idx=1,
    )

    entry = config["videos"][0]
    points = entry["primary_points_by_frame"]["0"]
    assert len(points) >= 2
    # No point should be inside person 1's bbox center region
    for p in points:
        assert not (400 <= p[0] <= 500 and 100 <= p[1] <= 400)


def test_bbox_to_points_tall_person():
    """Tall bounding boxes get multiple points."""
    from clear_stage.generate_points_config import bbox_to_points

    points = bbox_to_points([100, 50, 300, 600])
    assert len(points) >= 2


def test_bbox_to_points_small_person():
    """Small bounding boxes get one center point."""
    from clear_stage.generate_points_config import bbox_to_points

    points = bbox_to_points([100, 100, 200, 250])
    assert len(points) == 1
    assert points[0] == [150, 175]


def test_generate_config_uses_correct_frame_idx():
    """Config should use the specified frame index for points."""
    from clear_stage.generate_points_config import generate_config

    detections = [
        {"bbox": [100, 100, 200, 300], "score": 0.9},
        {"bbox": [400, 100, 500, 300], "score": 0.8},
    ]

    config = generate_config(
        video_path="test.mp4", output_dir="out",
        detections=detections, principal_idx=0, frame_idx=150,
    )

    entry = config["videos"][0]
    assert "150" in entry["primary_points_by_frame"]
    assert "0" not in entry["primary_points_by_frame"]
