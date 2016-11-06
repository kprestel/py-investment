import json
import os
from scrapy.exceptions import DropItem


# from crawler import settings

class JsonItemPipeline(object):
    def __init__(self):
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), '..', 'financials')):
            os.mkdir(os.path.join(os.path.dirname(__file__), '..', 'financials'))
            self.base_path = os.path.join(os.path.dirname(__file__), '..', 'financials')
            print(self.base_path)
        else:
            self.base_path = os.path.join(os.path.dirname(__file__), '..', 'financials')

    def process_item(self, item, spider):
        """
        :param item: scrapy item
        :param spider: scrapy spider
        :return: scrapy item

        this method creates the file path required as well as writes out the JSON to a file to later access

        the required file path is as follows
        {project root}
            -financials/
                -int(fiscal_year) i.e. 2016
                    -10-Q
                        -Q(int)_SYMBOL.json i.e. Q1_AAPL.json
                    -10-K
                        -FY_SYMBOL.json i.e. FY_AAPL.json
        """
        if not os.path.isdir(os.path.join(self.base_path, item['symbol'])):
            # dont bother checking if the rest of the paths exist, just make them
            os.mkdir(os.path.join(self.base_path, item['symbol']))
            item_base_path = os.path.join(self.base_path, item['symbol'])
            os.mkdir(os.path.join(item_base_path, str(item['fiscal_year'])))
            year_dir = os.path.join(item_base_path, str(item['fiscal_year']))
            os.mkdir(os.path.join(year_dir, '10-Q'))
            quarter_dir = os.path.join(year_dir, '10-Q')
            os.mkdir(os.path.join(year_dir, '10-K'))
            annual_dir = os.path.join(year_dir, '10-K')
        else:
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
        return item

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass
