import logging
import os
import sys

this_folder = os.path.dirname(__file__)
logs_position = os.path.join(this_folder, '../../test.log')


class LogFormatter(logging.Formatter):

    # Define colors
    skyblue = "\x1b[38;5;117m"
    yellow = "\x1b[33;20m"
    orange = "\x1b[38;5;214m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32m"
    reset = "\x1b[0m"
    
    # Log format string without function name for normal logs
    _format = "[%(asctime)s] | %(levelname)s | %(message)s"
    
    # Log format string with function name highlighted for error logs
    error_format = "[%(asctime)s] | %(levelname)s | " + bold_red + "[%(funcName)s]" + reset + ": %(message)s"

    # Custom level name formats with padding for alignment
    LEVELNAME_FORMATS = {
        logging.DEBUG: yellow + "%(levelname)-7s" + reset,
        logging.INFO: skyblue + "%(levelname)-7s" + reset,
        logging.WARNING: orange + "%(levelname)-7s" + reset,
        logging.ERROR: bold_red + "%(levelname)-7s" + reset,
        logging.CRITICAL: bold_red + "%(levelname)-7s" + reset
    }
    
    # Custom message formats
    MESSAGE_FORMATS = {
        logging.DEBUG: "%(message)s",
        logging.INFO: green + "%(message)s" + reset,  # Apply green to INFO messages
        logging.WARNING: "%(message)s",
        logging.ERROR: "%(message)s",
        logging.CRITICAL: "%(message)s"
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Determine if the log is an error or critical level, then use the special format
        if record.levelno in (logging.ERROR, logging.CRITICAL):
            log_fmt = self.error_format
        else:
            log_fmt = self._format
        
        # Replace %(levelname)s and %(message)s with the colored versions
        log_fmt = log_fmt.replace("%(levelname)s", self.LEVELNAME_FORMATS.get(record.levelno, "%(levelname)-7s"))
        log_fmt = log_fmt.replace("%(message)s", self.MESSAGE_FORMATS.get(record.levelno, "%(message)s"))
        
        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)

class Logs:
    
    tag: str = ""
    
    def __init__(
            self,
            loggername: str
    ):
        self.loggername = loggername
        self.file = logs_position

    def log(self, level=logging.INFO):

        logger = logging.getLogger(self.loggername)
        # Prevent adding multiple handlers to the logger
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(LogFormatter())
            logger.addHandler(handler)
            logger.propagate = False

        return logger
