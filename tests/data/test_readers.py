# noinspection PyUnresolvedReferences
import pytest
import pytech.data.reader as reader
import datetime as dt

def test_get_data():
    test = reader.get_data('GOOG')
    for k, v in test.items():
        print(f'k:{k}, v:{v}')
