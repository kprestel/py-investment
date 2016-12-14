from contextlib import contextmanager
from functools import wraps
from pytech import Session

@contextmanager
def query_session():
    session = Session()
    yield session
    session.close()

@contextmanager
def transactional_session(auto_close=True):
    """
    Helper method to manage db transactions

    :param auto_close: whether or not the session should be closed after it goes out of scope
    :type auto_close: bool
    :return: session
    """

    session = Session()
    # session.begin(nested=nested)
    try:
        yield session
    except:
        session.rollback()
        raise
    else:
        session.commit()
        if auto_close:
            session.close()

def in_transaction(**session_kwargs):
    """Decorator which wraps the decorated function in a transactional session. If the
       function completes successfully, the transaction is committed. If not, the transaction
       is rolled back."""
    def outer_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with transactional_session(**session_kwargs) as session:
                return func(session, *args, **kwargs)
        return wrapper
    return outer_wrapper