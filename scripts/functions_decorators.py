"""
Decorator functions for this program
"""

###############################################################################
# Import Python Libraries
###############################################################################
import logging; logging.basicConfig(level=logging.INFO)
from functools import wraps
from datetime import datetime

def my_timeit(f):                                                        
    """
    Decorator function to log function name and duration.
    Args:
        f:

    Returns:
        

    """
    @wraps(f)   # see docs. retains info about f                                
    def wrapped(*args, **kwargs):                                               
        logging.info(f'Starting function {f.__name__}')                         
        start = datetime.now()                                                  
        response = f(*args, **kwargs)                                           
        duration = (datetime.now() - start).total_seconds()                     
        logging.info(f'{f.__name__} finished.  Duration => {duration}')
        logging.info('\n\n')
        return response                                                         
    return wrapped


