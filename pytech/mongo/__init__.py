from arctic import Arctic, register_library_type
from pymongo import MongoClient

from pytech.mongo.barstore import BarStore
from pytech.mongo.portfolio_store import PortfolioStore


client = MongoClient('localhost')
ARCTIC_STORE = Arctic(client)

register_library_type(BarStore.LIBRARY_TYPE, BarStore)
register_library_type(PortfolioStore.LIBRARY_TYPE, PortfolioStore)

ARCTIC_STORE.initialize_library(BarStore.LIBRARY_NAME, BarStore.LIBRARY_TYPE)
ARCTIC_STORE.initialize_library(PortfolioStore.LIBRARY_NAME,
                                PortfolioStore.LIBRARY_TYPE)
