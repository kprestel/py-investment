# noinspection PyUnresolvedReferences
import pytest

from pytech.sources.restclient import RestClient, HTTPAction

def test_http_action():
    action = HTTPAction.check_if_valid('GET').name
    assert action == 'GET'

class TestRestClient(object):
    pass

