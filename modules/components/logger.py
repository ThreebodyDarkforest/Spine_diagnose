import logging, os
import datetime
from . import PATH

def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    nw_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    handler = logging.FileHandler(filename=os.path.join(PATH, 'logs', f'{nw_time}_log.txt'))
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(lineno)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

LOGGER = set_logging(__name__)