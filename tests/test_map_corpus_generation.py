from __future__ import annotations

import pytest

from generate_map_corpus import expand_regions


def config(size_jitter: float):
    return {
        "regions": [
            {
                "name": "varied",
                "center": [40.0, -120.0],
                "jitter": [0.01, 0.01],
                "half_size": [0.02, 0.03],
                "half_size_jitter": size_jitter,
                "subareas": 3,
                "split": "train",
            }
        ]
    }


def test_multiscale_regions_are_reproducible_and_not_identical():
    left = expand_regions(config(0.25), seed=7)
    right = expand_regions(config(0.25), seed=7)
    assert left == right
    heights = {round(item["bbox"][2] - item["bbox"][0], 6) for item in left}
    widths = {round(item["bbox"][3] - item["bbox"][1], 6) for item in left}
    assert len(heights) > 1
    assert len(widths) > 1


def test_invalid_size_jitter_is_rejected():
    with pytest.raises(ValueError, match="half_size_jitter"):
        expand_regions(config(1.0), seed=7)
