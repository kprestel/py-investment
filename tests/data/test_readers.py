# noinspection PyUnresolvedReferences
import pytest

from pytech.data.reader import BarReader


def test_get_data():
    reader = BarReader('pytech.bars')
    test = reader.get_data('GOOG')
    for k, v in test.items():
        print(f'k:{k}, v:{v}')
