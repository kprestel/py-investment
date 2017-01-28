import logging
from builtins import object
from datetime import date, timedelta

from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from pytech import utils
from pytech.crawler.loaders import ReportItemLoader

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
        super().__init__(**kwargs)

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

        self.start_date = utils.parse_date(start_date).strftime('%Y%m%d')
        self.end_date = utils.parse_date(end_date).strftime('%Y%m%d')

        # limit_arg = kwargs.get('limit', '')
        start = int(kwargs.get('start', 0))
        count = kwargs.get('count', None)
        if count is not None:
            count = int(count)

        if self.symbols_arg:
            self.start_urls = URLGenerator(self.symbols_arg, start_date=self.start_date, end_date= self.end_date,
                                               start=start, count=count)
            # else:
            # symbols = [self.symbols_arg]
            # self.start_urls = URLGenerator(symbols, start_date=self.start_date, end_date=self.end_date,
            #                                    start=start, count=count)
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
