import numpy as np
import pytest

from immich2frigate.frigate018 import (
    arcface_preprocess,
    blur_confidence_reduction,
    build_class_mean,
    reported_confidence,
    require_target,
    similarity_to_confidence,
)


def test_arcface_preprocess_converts_bgr_to_rgb_and_normalizes() -> None:
    # The blue BGR channel must land in channel 2 of Frigate's RGB tensor.
    bgr = np.zeros((16, 16, 3), dtype=np.uint8)
    bgr[:, :] = (255, 0, 0)

    result = arcface_preprocess(bgr)

    assert result.shape == (1, 3, 112, 112)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, :, 56, 56], [-1.0, -1.0, 1.0])


def test_arcface_preprocess_centers_aspect_fitted_image_on_black() -> None:
    bgr = np.full((16, 32, 3), 127, dtype=np.uint8)

    result = arcface_preprocess(bgr)

    assert np.all(result[0, :, :28, :] == -1.0)
    assert np.all(result[0, :, 84:, :] == -1.0)


def test_small_class_uses_per_dimension_trim_mean_without_outlier_filter() -> None:
    embeddings = [np.array([1.0, 0.0]) for _ in range(3)] + [np.array([-1.0, 0.0])]

    result = build_class_mean(embeddings)

    np.testing.assert_allclose(result, [0.5, 0.0])


def test_large_class_drops_whole_vector_outliers_before_trim_mean() -> None:
    embeddings = [np.array([1.0, 0.0]) for _ in range(7)] + [np.array([-1.0, 0.0]) for _ in range(3)]

    result = build_class_mean(embeddings)

    np.testing.assert_allclose(result, [1.0, 0.0])


def test_class_mean_preserves_arcface_float32_dtype() -> None:
    embeddings = [np.array([1.0, 0.0], dtype=np.float32) for _ in range(4)]

    result = build_class_mean(embeddings)

    assert result.dtype == np.float32


def test_confidence_and_blur_penalty_match_frigate_one_shot_score() -> None:
    assert similarity_to_confidence(0.3) == pytest.approx(0.5)
    assert blur_confidence_reduction(119.99) == 0.06
    assert blur_confidence_reduction(250) == 0.0
    assert blur_confidence_reduction(1, enabled=False) == 0.0
    assert reported_confidence(0.3, 119.99) == 0.44


def test_profile_rejects_unverified_versions_and_models() -> None:
    require_target("0.18.0", "large")
    require_target("0.18.0-77a66e75c618", "large")
    with pytest.raises(ValueError):
        require_target("0.18.0-deadbeef", "large")
    with pytest.raises(ValueError):
        require_target("0.18.1", "large")
    with pytest.raises(ValueError):
        require_target("0.18.0", "small")
