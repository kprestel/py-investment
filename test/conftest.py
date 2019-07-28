import os
import queue
from typing import Dict
from unittest.mock import MagicMock as Mock

import pandas as pd
import pytest
from dateutil.parser import parse
from dateutil.tz import tzutc

import pytech.trading.blotter as b
import pytech.utils as utils
from pytech import TEST_DATA_DIR
from pytech.data import BarReader
from pytech.data.handler import Bars
from pytech.fin.asset.asset import Stock
from pytech.fin.portfolio import BasicPortfolio
from pytech.fin.portfolio.handler import BasicSignalHandler
from pytech.trading.controls import MaxOrderCount
from pytech.utils import DateRange
import pytech.data.schema as schema
from testing.postgresql import Postgresql


@pytest.fixture()
def mock_env(monkeypatch):
    if os.getenv("TIINGO_API_KEY") is None:
        monkeypatch.setenv("TIINGO_API_KEY", "API_KEY")
    if os.getenv("ALPHA_VANTAGE_API_KEY") is None:
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "API_KEY")
    if os.getenv("BARCHART_API_KEY") is None:
        monkeypatch.setenv("BARCHART_API_KEY", "API_KEY")


@pytest.fixture(scope='session')
def database_uri(pg_server):
    print(pg_server)
    with Postgresql(
            postgres_args='-c TimeZone=UTC '
                          '-c fsync=off '
                          '-c synchronous_commit=off '
                          '-c full_page_writes=off '
                          '-c checkpoint_timeout=30min '
                          '-c bgwriter_delay=10000ms') as pgdb:
        yield pgdb.url()


@pytest.fixture(scope='session', autouse=True)
def docker_db(docker_db):
    return docker_db

@pytest.fixture()
def init_db(postgresql_db):
    engine = postgresql_db.engine
    schema.trade_actions.create(bind=engine)
    schema.order_types.create(bind=engine)
    schema.order_sub_types.create(bind=engine)
    postgresql_db.create_table(schema.assets)
    postgresql_db.create_table(schema.orders)
    postgresql_db.create_table(schema.trade)
    postgresql_db.create_table(schema.bars)
    postgresql_db.create_table(schema.portfolio)
    postgresql_db.create_table(schema.transaction)
    postgresql_db.create_table(schema.ownership)
    postgresql_db.create_table(schema.asset_snapshot)
    postgresql_db.create_table(schema.portfolio_snapshot)
    postgresql_db.install_extension('timescaledb')

@pytest.fixture(scope="module")
def vcr_config():
    return {
        'filter_query_parameters': ('apikey', 'XXXXXX'),
        'filter_headers': ('Authorization', 'XXXXXXXX')
    }


@pytest.fixture
def vcr_cassette_path(request, vcr_cassette_name):
    """
    Sets the cassette path.

    It is expected that the tests are being run
    from the project root dir.
    """
    return os.path.join('test', 'cassettes', request.module.__name__,
                        str(request.param_index), vcr_cassette_name)


@pytest.fixture(scope='session')
def start_date():
    return '2012-01-01'


@pytest.fixture(scope='session')
def end_date():
    return '2017-12-01'


@pytest.fixture(scope='session')
def date_range(start_date, end_date):
    return DateRange(start_date, end_date)


@pytest.fixture(scope='session')
def _ticker_df_cache():
    """
    Create a dictionary of dfs so we only have to go to disk once.
    """
    df_cache = {}
    for f in os.listdir(TEST_DATA_DIR):
        df = pd.read_csv(os.path.join(TEST_DATA_DIR, f),
                         index_col=utils.DATE_COL,
                         parse_dates=['date'])
        df.index.name = utils.DATE_COL
        df_cache[os.path.splitext(os.path.basename(f))[0]] = df
    return df_cache


# @pytest.fixture(autouse=True)
def write_ref_csv(monkeypatch, date_range):
    """
    This is a utils fixture that shouldn't be used unless generating
    reference data.
    """

    def to_csv(bar_reader: BarReader, tickers,
               source='google',
               start=start_date,
               end=end_date,
               check_db=True,
               filter_data=True,
               **kwargs):
        start = utils.parse_date(start)
        end = utils.parse_date(end)
        _ = kwargs.pop('columns', None)
        if isinstance(tickers, str):
            df = bar_reader._single_get_data(tickers,
                                             source,
                                             date_range,
                                             check_db,
                                             **kwargs)
            df.df.to_csv(f'{TEST_DATA_DIR}{os.sep}{tickers}.csv')
            return df.df
        else:
            if isinstance(tickers, pd.DataFrame):
                tickers = tickers.index
            df = bar_reader._mult_tickers_get_data(tickers,
                                                   source,
                                                   date_range,
                                                   check_db,
                                                   **kwargs)
            for ticker, df_ in df.items():
                df_.to_csv(f'{TEST_DATA_DIR}{os.sep}{ticker}.csv')
            return df

    monkeypatch.setattr(BarReader, 'get_data', to_csv)


# @pytest.fixture(autouse=True)
def no_db(monkeypatch, _ticker_df_cache: Dict[str, pd.DataFrame]):
    """Don't make any database calls. Read all data from `TEST_DATA_DIR`"""

    def patch_get_data(bar_reader, tickers, *args, **kwargs):
        if isinstance(tickers, str):
            return _ticker_df_cache[tickers]

        dfs = {}
        for ticker in tickers:
            df = _ticker_df_cache[ticker]
            dfs[ticker] = df

        return dfs

    monkeypatch.setattr(BarReader, 'get_data', patch_get_data)


@pytest.fixture(scope='session')
def aapl_df():
    """Returns a OHLCV df for Apple."""
    return pd.read_csv(f'{TEST_DATA_DIR}{os.sep}AAPL.csv')


@pytest.fixture(scope='session')
def fb_df():
    """Returns a OHLCV df for Apple."""
    return pd.read_csv(f'{TEST_DATA_DIR}{os.sep}FB.csv')


def date_utc(s):
    return parse(s, tzinfos=tzutc)


@pytest.fixture(scope='session')
def cvs_df():
    return pd.read_csv(f'{TEST_DATA_DIR}{os.sep}CVS.csv')


@pytest.fixture(scope='session')
def goog_df():
    return pd.read_csv(f'{TEST_DATA_DIR}{os.sep}GOOG.csv')


def get_test_csv_path(ticker):
    """Return the path to the test CSV file"""
    return f'{TEST_DATA_DIR}{os.sep}{ticker}.csv'


@pytest.fixture(scope='session')
def ticker_list():
    return {'AAPL', 'MSFT', 'CVS', 'FB'}
    # return {'MSFT'}


@pytest.fixture(scope='session')
def events():
    return queue.Queue()


@pytest.fixture()
def mock_portfolio():
    """A mock portfolio that does nothing but be a mock."""
    return Mock(spec=BasicPortfolio)


@pytest.fixture()
def bars(events, ticker_list, date_range):
    """Create a default :class:`YahooDataHandler`"""
    bars = Bars(events, ticker_list, date_range)
    bars.update_bars()
    return bars


@pytest.fixture()
def blotter(events, bars):
    return b.Blotter(events, bars=bars)


@pytest.fixture()
def populated_blotter(blotter: b.Blotter, mock_portfolio, start_date):
    """Populate the blot and return it."""
    blotter.controls.append(MaxOrderCount(True, 10))
    blotter.current_dt = start_date

    blotter.place_order(mock_portfolio, 'AAPL', 50, 'BUY', 'LIMIT',
                        limit_price=100.10,
                        order_id='one')
    blotter.place_order(mock_portfolio, 'AAPL', 50, 'BUY', 'LIMIT',
                        limit_price=98.10,
                        order_id='two')
    blotter.place_order(mock_portfolio, 'MSFT', 50, 'SELL', 'LIMIT',
                        limit_price=93.10,
                        order_id='three')
    blotter.place_order(mock_portfolio, 'FB', 50, 'SELL', 'LIMIT',
                        limit_price=105.10,
                        order_id='four')

    return blotter


@pytest.fixture()
def basic_portfolio(events, bars, date_range, populated_blotter):
    """Return a BasicPortfolio to be used in testing."""
    populated_blotter.bars = bars
    return BasicPortfolio(bars, events, date_range, populated_blotter)


@pytest.fixture()
def empty_portfolio(events, date_range, blotter):
    return BasicPortfolio(blotter.bars, events, date_range, blotter)


@pytest.fixture()
def basic_signal_handler(basic_portfolio):
    """Return a BasicSignalHandler to be used in testing."""
    return BasicSignalHandler()


@pytest.fixture(scope='session')
def aapl(date_range):
    return Stock('AAPL', date_range)


@pytest.fixture(scope='session')
def fb(date_range):
    return Stock('FB', date_range)
