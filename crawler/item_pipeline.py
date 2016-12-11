import json
from dateutil import parser
from dateutil.relativedelta import relativedelta
from datetime import datetime
import os

from scrapy import Selector
from scrapy.exceptions import DropItem
import logging
# from pytech import Session
import pytech.db_utils as db
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
        logger.info('{} opened'.format(self.__class__))


    def process_item(self, item, spider):
        """
        Process Scrapy items

        :param item: scrapy item
        :param spider: scrapy spider
        :return:
        """
        with db.transactional_session() as session:
            logger.info('Creating fundamental obj for ticker: {}'.format(item['symbol']))
            access_key = str(item.get('fiscal_year')) + '_' + item.get('period_focus')

            q = session.query(Fundamental).filter(Fundamental.access_key == access_key).first()
            if q is not None:
                logger.warning('Dropped item because access_key: {} exists'.format(access_key))
                raise DropItem
            fundamental_dict = {}
            fundamental_dict['amended'] = item.get('amend')
            fundamental_dict['assets'] = item.get('assets')
            fundamental_dict['current_assets'] = item.get('cur_assets')
            fundamental_dict['current_liabilities'] = item.get('cur_liab')
            fundamental_dict['cash'] = item.get('cash')
            fundamental_dict['dividend'] = item.get('dividend')
            fundamental_dict['end_date'] = item.get('end_date')
            fundamental_dict['eps'] = item.get('eps_basic')
            fundamental_dict['eps_diluted'] = item.get('eps_diluted')
            fundamental_dict['equity'] = item.get('equity')
            fundamental_dict['net_income'] = item.get('net_income')
            fundamental_dict['operating_income'] = item.get('op_income')
            fundamental_dict['revenues'] = item.get('revenues')
            fundamental_dict['investment_revenues'] = item.get('investment_revenues')
            fundamental_dict['fin_cash_flow'] = item.get('cash_flow_fin')
            fundamental_dict['inv_cash_flow'] = item.get('cash_flow_inv')
            fundamental_dict['ops_cash_flow'] = item.get('cash_flow_op')
            fundamental_dict['period_focus'] = item.get('period_focus')
            fundamental_dict['year'] = item.get('fiscal_year')
            fundamental_dict['property_plant_equipment'] = item.get('property_plant_equipment')
            fundamental_dict['gross_profit'] = item.get('gross_profit')
            fundamental_dict['tax_expense'] = item.get('tax_expense')
            fundamental_dict['net_taxes_paid'] = item.get('net_taxes_paid')
            fundamental_dict['acts_receive_current'] = item.get('acts_receive_current')
            fundamental_dict['acts_pay_current'] = item.get('acts_pay_current')
            fundamental_dict['accrued_liabilities_current'] = item.get('accrued_liabilities_current')
            fundamental_dict['acts_receive_noncurrent'] = item.get('acts_receive_noncurrent')
            fundamental_dict['inventory_net'] = item.get('inventory_net')
            fundamental_dict['interest_expense'] = item.get('interest_expense')
            fundamental_dict['total_liabilities'] = item.get('total_liabilities')
            fundamental_dict['total_liabilities_equity'] = item.get('total_liabilities_equity')
            fundamental_dict['shares_outstanding'] = item.get('shares_outstanding')
            fundamental_dict['shares_outstanding_diluted'] = item.get('shares_outstanding_diluted')
            fundamental_dict['common_stock_outstanding'] = item.get('common_stock_outstanding')
            fundamental_dict['depreciation_amortization'] = item.get('deprecation_amortization')
            fundamental_dict['cogs'] = item.get('cogs')
            fundamental_dict['comprehensive_income_net_of_tax'] = item.get('comprehensive_income_net_of_tax')
            fundamental_dict['research_and_dev_expense'] = item.get('research_and_dev_expense')
            fundamental_dict['warranty_accrual'] = item.get('warranty_accrual')
            fundamental_dict['warranty_accrual_payments'] = item.get('warranty_accrual_payments')
            fundamental_dict['ticker'] = item.get('symbol')

            logger.info('Created Fundamental obj for ticker: {}'.format(item['symbol']))
            session.add(Fundamental.from_dict(fundamental_dict=fundamental_dict))
            return item

    def close_spider(self, spider):
        logger.info('{} closing.'.format(self.__class__))
        pass
        # self.session.commit()
        # self.session.close()
