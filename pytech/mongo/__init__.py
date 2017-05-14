from arctic import Arctic, register_library_type
from pytech.mongo.barstore import BarStore
from pytech.mongo.portfolio_store import PortfolioStore

ARCTIC_STORE = Arctic('localhost')

register_library_type(BarStore.LIBRARY_TYPE, BarStore)
register_library_type(PortfolioStore.LIBRARY_TYPE, PortfolioStore)

ARCTIC_STORE.initialize_library(BarStore.LIBRARY_NAME, BarStore.LIBRARY_TYPE)
ARCTIC_STORE.initialize_library(PortfolioStore.LIBRARY_NAME,
                                PortfolioStore.LIBRARY_TYPE)



