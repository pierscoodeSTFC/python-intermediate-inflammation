"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean
from inflammation.models import daily_min
from inflammation.models import daily_max

@pytest.mark.parametrize(
"test_input, expected_result",
[
    ([[0, 0], [0, 0], [0, 0]], [0, 0]),
    ([[1, 2], [3, 4], [5, 6]], [3, 4])
]
)
def test_daily_mean(test_input, expected_result):
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), expected_result)


@pytest.mark.parametrize(
"test_input, expected_result",
[
    ([[0, 0], [0, 0], [0, 0]], [0, 0]),
    ([[1, 6], [3, 4], [5, 2]], [5, 6])
]
)
def test_daily_max(test_input, expected_result):
    """Test that the max value works for an array of integers."""
    from inflammation.models import daily_max

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), expected_result)


@pytest.mark.parametrize(
"test_input, expected_result",
[
    ([[0, 0], [0, 0], [0, 0]], [0, 0]),
    ([[1, 4], [3, 2], [5, 6]], [1, 2])
]
)
def test_daily_min(test_input, expected_result):
    """Test that the max value works for an array of integers."""
    from inflammation.models import daily_min

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), expected_result)
