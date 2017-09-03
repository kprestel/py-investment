"""
This module is for lightweight data holders to make interfacing the
return values these functions easier.
"""
from collections import namedtuple

DfLibName = namedtuple('DfLibName', ['df', 'lib_name'])
