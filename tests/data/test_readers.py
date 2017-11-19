# noinspection PyUnresolvedReferences
import pytest

from pytech.data.reader import BarReader


def test_get_data(date_range):
    reader = BarReader('pytech.bars')
    test = reader.get_data('GOOG', date_range=date_range)
    for k, v in test.items():
        print(f'k:{k}, v:{v}')
