import pytest
import pytech.utils.common_utils as com_utils


def test_iterable_to_set():
    """Test and make sure that a set is returned."""

    test_iterable = ['AAPL', 'MSFT', 'AAPL', 'FB', 'MSFT']
    returned_set = com_utils.iterable_to_set(test_iterable)
    assert len(returned_set) == 3

    test_iterable = (1, 2, 3)
    returned_set = com_utils.iterable_to_set(test_iterable)
    assert len(returned_set) == 3

    with pytest.raises(TypeError):
        com_utils.iterable_to_set(12)
