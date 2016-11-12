# Scrapy settings for pyTech project
#
# adapted from pystock-crawler project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'pytech_crawler'

EXPORT_FIELDS = (
    # Price columns
    'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close',

    # Report columns
    'end_date', 'amend', 'period_focus', 'fiscal_year', 'doc_type', 'revenues', 'op_income', 'net_income',
    'eps_basic', 'eps_diluted', 'dividend', 'assets', 'cur_assets', 'cur_liab', 'cash', 'equity',
    'cash_flow_op', 'cash_flow_inv', 'cash_flow_fin',
)

FEED_EXPORTERS = {
    'csv': 'crawler.exporters.CsvItemExporter2',
    'symbollist': 'crawler.exporters.SymbolListExporter'
}

HTTPCACHE_ENABLED = True

HTTPCACHE_POLICY = 'scrapy.extensions.httpcache.RFC2616Policy'

# THIS IS BAD BECAUE IT CAUSES IO EXCEPTIONS
# HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.LeveldbCacheStorage'

HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.DbmCacheStorage'

LOG_LEVEL = 'DEBUG'

NEWSPIDER_MODULE = 'crawler.spiders'

SPIDER_MODULES = ['crawler.spiders']

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'pystock-crawler (+http://www.yourdomain.com)'

CONCURRENT_REQUESTS_PER_DOMAIN = 8

COOKIES_ENABLED = False

#AUTOTHROTTLE_ENABLED = True

RETRY_TIMES = 4

EXTENSIONS = {
    'scrapy.throttle.AutoThrottle': None,
    'crawler.throttle.PassiveThrottle': 0
}

PASSIVETHROTTLE_ENABLED = True
#PASSIVETHROTTLE_DEBUG = True

DEPTH_STATS_VERBOSE = True

ITEM_PIPELINES = {
    'crawler.item_pipeline.JsonItemPipeline': 800
}

# BASE_OUTPUT_FILE_PATH = ''
