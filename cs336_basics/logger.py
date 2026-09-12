import logging

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that uses tqdm.write() to avoid interfering with progress bars.

    Registered as the console handler in Hydra's job-logging config
    (configs/hydra/job_logging/tqdm.yaml), so Hydra owns logging setup while log records still
    route through tqdm.write() and don't clobber progress bars.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)
