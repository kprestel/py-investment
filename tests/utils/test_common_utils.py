import pytest
import pytech.utils as utils


def test_iterable_to_set():
    """Test and make sure that a set is returned."""

    test_iterable = ['AAPL', 'MSFT', 'AAPL', 'FB', 'MSFT']
    returned_set = utils.iterable_to_set(test_iterable)
    assert len(returned_set) == 3

    test_iterable = (1, 2, 3)
    returned_set = utils.iterable_to_set(test_iterable)
    assert len(returned_set) == 3

    with pytest.raises(TypeError):
        utils.iterable_to_set(12)

def test_borg():
    """Test and make sure that a class acts like a singleton."""

    class _TestBorg(utils.Borg):
        """Test borg class."""

        def __init__(self, test='foo'):
            super().__init__()
            self.test = test

    borg = _TestBorg()
    assert borg.test == 'foo'
    borg2 = _TestBorg('bar')
    assert borg2.test == 'bar'
    assert borg.test == 'bar'


@pytest.mark.parametrize('input, expected',
                          [('foo', False),
                           (['foo', 'bar'], True),
                           (12, False),
                           (('xyz',), True)])
def test_is_iterable(input, expected):
    assert utils.is_iterable(input) == expected

