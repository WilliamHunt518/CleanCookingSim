from __future__ import annotations

import numpy as np
import pytest

from grid_energy.resample import resample_kw, tile_to_length


def test_downsample_averages_consecutive_blocks():
    # 6 blocks of 5-min -> 2 blocks of 15-min, averaging each group of 3
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = resample_kw(series, from_block_minutes=5.0, to_block_minutes=15.0)
    assert np.allclose(out, [2.0, 5.0])  # mean(1,2,3)=2, mean(4,5,6)=5


def test_upsample_repeats_each_block():
    series = np.array([10.0, 20.0])
    out = resample_kw(series, from_block_minutes=15.0, to_block_minutes=5.0)
    assert np.allclose(out, [10.0, 10.0, 10.0, 20.0, 20.0, 20.0])


def test_same_resolution_is_a_noop():
    series = np.array([1.0, 2.0, 3.0])
    out = resample_kw(series, from_block_minutes=15.0, to_block_minutes=15.0)
    assert np.allclose(out, series)


def test_non_integer_ratio_raises():
    with pytest.raises(ValueError):
        resample_kw(np.zeros(10), from_block_minutes=7.0, to_block_minutes=15.0)


def test_downsample_length_not_divisible_raises():
    with pytest.raises(ValueError):
        resample_kw(np.zeros(5), from_block_minutes=5.0, to_block_minutes=15.0)  # 5 not divisible by 3


def test_tile_to_length_repeats_and_truncates():
    series = np.array([1.0, 2.0, 3.0])
    out = tile_to_length(series, target_len=7)
    assert np.allclose(out, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])


def test_tile_to_length_truncates_when_already_long_enough():
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = tile_to_length(series, target_len=3)
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_tile_to_length_rejects_empty_series():
    with pytest.raises(ValueError):
        tile_to_length(np.array([]), target_len=5)
