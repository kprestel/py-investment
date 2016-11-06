import os
from builtins import object

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
        start_date = kwargs.get('startdate', '')
        end_date = kwargs.get('enddate', '')
        limit_arg = kwargs.get('limit', '')

        utils.check_date_arg(start_date, 'startdate')
        utils.check_date_arg(end_date, 'enddate')
        start, count = utils.parse_limit_arg(limit_arg)

        if symbols_arg:
            if os.path.exists(symbols_arg):
                # get symbols from a text file
                symbols = utils.load_symbols(symbols_arg)
            else:
                # inline symbols in command
                symbols = symbols_arg.split(',')
            self.start_urls = URLGenerator(symbols, start_date, end_date, start, count)
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
                print(item)
                return item
        return None

# from scrapy.crawler import CrawlerProcess
# from scrapy.crawler import CrawlerRunner
# from scrapy.utils.project import get_project_settings
# from twisted.internet import reactor
# # process = CrawlerProcess({'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'})
# runner = CrawlerRunner()
# process = CrawlerProcess(get_project_settings())
# spider = EdgarSpider(symbols='AAPL', startdate='20160101', enddate='20161104')
# d = runner.crawl(spider)
# # d.addBoth(lambda _: reactor.stop())
# # reactor.run()
# process.crawl(spider)
# process.start()
