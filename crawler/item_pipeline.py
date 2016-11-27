import json
from dateutil import parser
from dateutil.relativedelta import relativedelta
from datetime import datetime
import os

from scrapy import Selector
from scrapy.exceptions import DropItem
import logging
from pytech import Session
from pytech.stock import Stock, Fundamental

logger = logging.getLogger(__name__)


# from crawler import settings

class JsonItemPipeline(object):
    def __init__(self):
        logger.info('JsonItemPipeline starting processing')
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), '..', 'financials')):
            logger.info('File path not found...\nCreating new file path.')
            os.mkdir(os.path.join(os.path.dirname(__file__), '..', 'financials'))
            self.base_path = os.path.join(os.path.dirname(__file__), '..', 'financials')
            logger.debug('Base Path: {}'.format(self.base_path))
        else:
            self.base_path = os.path.join(os.path.dirname(__file__), '..', 'financials')
            logger.debug('Base Path: {}'.format(self.base_path))

    def process_item(self, item, spider):
        """
        :param item: scrapy item
        :param spider: scrapy spider
        :return: scrapy item

        this method creates the file path required as well as writes out the JSON to a file to later access

        the required file path is as follows
        {project root}
            -financials/
                -symbol
                    -int(fiscal_year) i.e. 2016
                        -10-Q
                            -Q(int)_SYMBOL.json i.e. Q1_AAPL.json
                        -10-K
                            -FY_SYMBOL.json i.e. FY_AAPL.json
        """

        if not os.path.isdir(os.path.join(self.base_path, item['symbol'])):
            # dont bother checking if the rest of the paths exist, just make them
            logger.info(
                'Existing base file path not found for symbol: {}...\nCreating it now...'.format(item['symbol']))
            os.mkdir(os.path.join(self.base_path, item['symbol']))
            item_base_path = os.path.join(self.base_path, item['symbol'])
            logger.debug('Item base file path: {}'.format(item['symbol']))
            os.mkdir(os.path.join(item_base_path, str(item['fiscal_year'])))
            year_dir = os.path.join(item_base_path, str(item['fiscal_year']))
            os.mkdir(os.path.join(year_dir, '10-Q'))
            quarter_dir = os.path.join(year_dir, '10-Q')
            os.mkdir(os.path.join(year_dir, '10-K'))
            annual_dir = os.path.join(year_dir, '10-K')
            logger.info('Created file structure for symbol: {}'.format(item['symbol']))
        else:
            logger.info('Existing base file path found for symbol: {}'.format(item['symbol']))
            item_base_path = os.path.join(self.base_path, item['symbol'])
            year_dir = os.path.join(item_base_path, str(item['fiscal_year']))
            print(item['fiscal_year'])
            quarter_dir = os.path.join(year_dir, '10-Q')
            annual_dir = os.path.join(year_dir, '10-K')
        if item['period_focus'] == 'FY':
            output_file = 'FY_{}.json'.format(item['symbol'])
            output_path = os.path.join(annual_dir, output_file)
        else:
            output_file = '{}_{}.json'.format(item['period_focus'], item['symbol'])
            output_path = os.path.join(quarter_dir, output_file)
        # check if the file exists before overwriting it
        if os.path.isfile(output_path):
            raise DropItem('{} already exists! existing file was not overwritten'.format(output_path))
        else:
            with open(output_path, 'w') as f:
                json.dump(dict(item), f)
                logger.info('JSON file for symbol: {} can be found at: {}'.format(item['symbol'], output_path))
        return item

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass


class FundamentalItemPipeline(object):
    """
    Create a Fundamental object and the corresponding stock (if it doesn't already exist) and add it to the db
    """

    def open_spider(self, spider):
        self.session = Session()

    def process_item(self, item, spider):
        logger.info('Creating fundamental obj for ticker: {}'.format(item['symbol']))
        fundamental_dict = {}
        fundamental_dict['amended'] = item['amend']
        fundamental_dict['assets'] = item['assets']
        fundamental_dict['current_assets'] = item['cur_assets']
        fundamental_dict['current_liabilities'] = item['cur_liab']
        fundamental_dict['cash'] = item['cash']
        fundamental_dict['dividend'] = item['dividend']
        fundamental_dict['end_date'] = item['end_date']
        fundamental_dict['eps'] = item['eps_basic']
        fundamental_dict['eps_diluted'] = item['eps_diluted']
        fundamental_dict['equity'] = item['equity']
        fundamental_dict['net_income'] = item['net_income']

        try:
            fundamental_dict['operating_income'] = item['op_income']
            # selector means XML tag
            if type(fundamental_dict['operating_income']) == Selector:
                fundamental_dict['operating_income'] = None
                logger.warning(
                    'operating income was of type {} so it could not be used'.format(type(item['op_income'])))
        except KeyError:
            logger.warning('op_income could not be found for {}'.format(item['symbol']))
            fundamental_dict['operating_income'] = None

        fundamental_dict['revenues'] = item['revenues']

        try:
            fundamental_dict['investment_revenues'] = item['investment_revenues']
            if type(fundamental_dict['investment_revenues']) == Selector:
                fundamental_dict['investment_revenues'] = None
                logger.warning('investment_revenues was of type {} so it could not be used' \
                               .format(type(item['investment_revenues'])))
        except KeyError:
            logger.warning('investment_revenues could not be found for {}'.format(item['symbol']))
            fundamental_dict['investment_revenues'] = None

        fundamental_dict['fin_cash_flow'] = item['cash_flow_fin']
        fundamental_dict['inv_cash_flow'] = item['cash_flow_inv']
        fundamental_dict['ops_cash_flow'] = item['cash_flow_op']
        fundamental_dict['period_focus'] = item['period_focus']
        fundamental_dict['year'] = item['fiscal_year']
        fundamental_dict['ticker'] = item['symbol']
        fundamental_dict['property_plant_equipment'] = item.get('property_plant_equipment')
        fundamental_dict['gross_profit'] = item.get('gross_profit')
        fundamental_dict['tax_expense'] = item.get('tax_expense')
        fundamental_dict['net_taxes_paid'] = item.get('net_taxes_paid')
        fundamental_dict['acts_receive_current'] = item.get('acts_receive_current')
        fundamental_dict['acts_pay_current'] = item.get('acts_pay_current')
        fundamental_dict['accrued_liabilities_current'] = item.get('accrued_liabilities_current')
        fundamental_dict['acts_receive_noncurrent'] = item.get('acts_receive_noncurrent')
        logger.info('Created Fundamental obj for ticker: {}'.format(item['symbol']))
        self.session.add(Fundamental.from_dict(fundamental_dict=fundamental_dict))
        return item

    def close_spider(self, spider):
        self.session.commit()
        self.session.close()


def get_newer_date(date_one, date_two):
    """
    :param date_one: str or datetime
    :param date_two: str or datetime
    :return: datetime

    returns the newer of two dates, as in the date that is closer to today
    """
    if type(date_one) != datetime:
        try:
            date_one = parser.parse(date_one)
        except ValueError:
            raise ValueError('could not convert date_one to datetime')
    if type(date_two) != datetime:
        try:
            date_two = parser.parse(date_two)
        except ValueError:
            raise ValueError('could not convert date_two to datetime')
    return max(date_one, date_two)


def get_older_date(date_one, date_two):
    """
    :param date_one: str or datetime
    :param date_two: str or datetime
    :return: datetime

    returns the older of two dates, as in the date that is further away from today
    """
    if type(date_one) != datetime:
        try:
            date_one = parser.parse(date_one)
            # date_one = datetime.strptime(date_one, '%Y-%m-%d')
        except ValueError:
            raise ValueError('could not convert date_one to datetime')
    if type(date_two) != datetime:
        try:
            date_two = parser.parse(date_two)
            # date_two = datetime.strptime(date_two, '%Y-%m-%d')
        except ValueError:
            raise ValueError('could not convert date_two to datetime')
    return min(date_one, date_two)


def guess_start_date(period_focus, end_date):
    """
    :param period_focus: str
    :param end_date: str or datetime
    :return: datetime

    attempts to best guess the start date by subtracting 3 months if the period_focus is quarterly and 12 months if it
    is a full year
    """
    end = parser.parse(end_date)
    if period_focus != 'FY':
        return end + relativedelta(months=-3)
    else:
        return end + relativedelta(months=-12)
