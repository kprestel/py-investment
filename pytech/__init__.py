import logging
import os
from os.path import dirname, join, pardir

PROJECT_DIR = dirname(__file__)
RESOURCE_DIR = join(pardir, 'resources')
DATA_DIR = join(RESOURCE_DIR, 'data')
TEST_DATA_DIR = join(pardir, 'tests', 'sample_data', 'csv')

try:
    os.makedirs(RESOURCE_DIR)
except OSError:
    pass

try:
    os.makedirs(DATA_DIR)
except OSError:
    pass

logging.basicConfig(level=logging.DEBUG)
