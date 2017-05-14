from arctic import Arctic, register_library_type
from pytech.mongo.barstore import BarStore

ARCTIC_STORE = Arctic('localhost')

register_library_type(BarStore._LIBRARY_TYPE, BarStore)

ARCTIC_STORE.initialize_library('pytech.bars', BarStore._LIBRARY_TYPE)



