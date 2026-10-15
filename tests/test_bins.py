"""D09--D13 bin regression tests."""

import math
import random

import numpy as np
import pytest

from thesis.geometry.bins import distance_bins, repeats_for_bin, sample_log_uniform_target


def test_locked_terminal_and_zero_gap_bounds():
    by_id = {item.bin_id: item for item in distance_bins()}

    assert (by_id[-5].lower_m, by_id[-5].upper_m) == pytest.approx((-5, -math.sqrt(15)))
    assert (by_id[80].lower_m, by_id[80].upper_m) == pytest.approx((math.sqrt(3200), 80))
    assert not by_id[-1].contains(-0.299999)
    assert not by_id[1].contains(0.299999)
    assert by_id[-1].contains(-0.3)
    assert by_id[1].contains(0.3)


def test_shared_boundaries_belong_to_one_bin_only():
    by_id = {item.bin_id: item for item in distance_bins()}

    assert not by_id[-5].contains(-math.sqrt(15))
    assert by_id[-3].contains(-math.sqrt(15))
    assert not by_id[1].contains(math.sqrt(3))
    assert by_id[3].contains(math.sqrt(3))
    assert by_id[80].contains(80)


def test_targets_are_log_uniform_and_signed():
    rng = random.Random(7)
    negative = [sample_log_uniform_target(distance_bins()[0], rng) for _ in range(4_000)]
    positive = [sample_log_uniform_target(distance_bins()[-1], rng) for _ in range(4_000)]

    assert all(value < 0 for value in negative)
    assert all(value > 0 for value in positive)
    assert np.mean(np.log(np.abs(negative))) == pytest.approx(np.mean(np.log([math.sqrt(15), 5])), abs=0.02)
    assert np.mean(np.log(positive)) == pytest.approx(np.mean(np.log([math.sqrt(3200), 80])), abs=0.02)


def test_near_bins_have_three_repeats():
    assert repeats_for_bin(-1) == 3
    assert repeats_for_bin(1) == 3
    assert all(repeats_for_bin(bin_id) == 1 for bin_id in (-5, -3, 3, 5, 10, 20, 40, 80))


def test_distance_bin_ids_are_ordered_and_unique():
    assert [item.bin_id for item in distance_bins()] == [-5, -3, -1, 1, 3, 5, 10, 20, 40, 80]
