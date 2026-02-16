import logging
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that uses tqdm.write() to avoid interfering with progress bars."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging to use tqdm-compatible handler.
    Returns the root logger configured with the TqdmLoggingHandler.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
