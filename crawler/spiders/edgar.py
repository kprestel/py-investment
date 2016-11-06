import os
from builtins import object

from datetime import datetime
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from crawler import utils
from crawler.loaders import ReportItemLoader


class URLGenerator(object):
    def __init__(self, symbols, start_date='', end_date='', start=0, count=None):
        # end = start + count if count is not None else None
        if count is not None:
            end = start + count
        else:
            end = None
        self.symbols = symbols[start:end]
        self.start_date = start_date
        self.end_date = end_date

    def __iter__(self):
        url = 'http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=10-&dateb={}&datea={}&owner=exclude&count=300'
        for symbol in self.symbols:
            yield (url.format(symbol, self.end_date, self.start_date))


class EdgarSpider(CrawlSpider):
    name = 'edgar'
    allowed_domains = ['www.sec.gov']

    rules = (Rule(LinkExtractor(allow=('/Archives/edgar/data/[^\"]+\-index\.htm'))),
             Rule(LinkExtractor(allow=('/Archives/edgar/data/[^\"]+/[A-Za-z]+\-\d{8}\.xml')), follow=True,
                  callback='parse_10qk'))

    def __init__(self, **kwargs):
        super(EdgarSpider, self).__init__(**kwargs)

        symbols_arg = kwargs.get('symbols')
        start_date = kwargs.get('start_date', '')
        end_date = kwargs.get('end_date', '')
        # limit_arg = kwargs.get('limit', '')
        start = int(kwargs.get('start', 0))
        count = kwargs.get('count', None)
        if count is not None:
            count = int(count)

        self.check_date_arg(start_date, 'start_date')
        self.check_date_arg(end_date, 'end_date')

        if symbols_arg:
            if isinstance(symbols_arg, list):
                self.start_urls = URLGenerator(symbols_arg, start_date=start_date, end_date=end_date, start=start, count=count)
            else:
                symbols = [symbols_arg]
                self.start_urls = URLGenerator(symbols, start_date=start_date, end_date=end_date, start=start, count=count)
            for s in self.start_urls:
                self.logger.info(s)
        else:
            self.start_urls = []


    def parse_10qk(self, response):
        """
        :param response: str, response from spider
        :return:

        Parse 10-Q or 10-K XML report.
        """
        self.logger.info('parsing {}'.format(response.url))
        loader = ReportItemLoader(response=response)
        item = loader.load_item()

        if 'doc_type' in item:
            doc_type = item['doc_type']
            if doc_type in ('10-Q', '10-K'):
                return item
        return None


    @classmethod
    def check_date_arg(cls, value, arg_name=None):
        if value:
            try:
                if len(value) != 8:
                    raise ValueError
                datetime.strptime(value, '%Y%m%d')
            except ValueError:
                raise ValueError("Option '{}' must be in YYYYMMDD format, input is '{}'".format(arg_name, value))

