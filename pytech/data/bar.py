class Bar(object):
    """
    Represents one *tick* of data. A *tick* can be any period of time.
    """
    def __init__(self, tick_frequency):
        self.tick_frequency = tick_frequency

