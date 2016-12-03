import os
from builtins import object
import logging
from dateutil import parser

from datetime import datetime, date
from datetime import timedelta
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from crawler.loaders import ReportItemLoader

logger = logging.getLogger(__name__)


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

        self.symbols_arg = kwargs.get('symbols')
        if 'start_date' in kwargs:
            start_date = kwargs.get('start_date')
        else:
            start_date = date.today() - timedelta(days=365)
            start_date = start_date
        if 'end_date' in kwargs:
            end_date = kwargs.get('end_date')
        else:
            end_date = date.today()

        try:
            start_date = parser.parse(start_date)
        except ValueError:
            raise ValueError('Error parsing start_date. {} was provided.'.format(start_date))
        except AttributeError:
            self.start_date = start_date.strftime('%Y%m%d')
        else:
            self.start_date = start_date.strftime('%Y%m%d')

        try:
            end_date = parser.parse(end_date)
        except ValueError:
            raise ValueError('Error parsing start_date. {} was provided.'.format(start_date))
        except AttributeError:
            self.end_date = end_date.strftime('%Y%m%d')
        else:
            self.end_date = end_date.strftime('%Y%m%d')

        # limit_arg = kwargs.get('limit', '')
        start = int(kwargs.get('start', 0))
        count = kwargs.get('count', None)
        if count is not None:
            count = int(count)

        if self.symbols_arg:
            if isinstance(self.symbols_arg, list):
                self.start_urls = URLGenerator(self.symbols_arg, start_date=self.start_date, end_date= self.end_date,
                                               start=start, count=count)
            else:
                symbols = [self.symbols_arg]
                self.start_urls = URLGenerator(symbols, start_date=self.start_date, end_date=self.end_date,
                                               start=start, count=count)
            for s in self.start_urls:
                logger.info('Start URL: {}'.format(s))
        else:
            logger.warning('No start URLs created!')
            self.start_urls = []
        logger.info('spider created!')



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
                self.logger.info('{} found, returning as item'.format(doc_type))
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

