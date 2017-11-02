from os import path, remove
import logging
import logging.config
import json

from .lstm import RunLSTM
from .data_generate import DataGenerator
from .model_callbacks import Checkpoints, Histories
from .utils import Metrics, ThreadSafe

# Source : http://www.patricksoftwareblog.com/python-logging-tutorial/

# Check if log file already exists. If Yes, then remove it for fresh writing.
if path.isfile("src/logs/debug.log") and path.isfile("src/logs/errors.log"):
    remove("src/logs/debug.log")
    remove("src/logs/errors.log")

# Load log configuration file
with open("log_config.json", 'r') as log_config_file:
    config_dict = json.load(log_config_file)

logging.config.dictConfig(config_dict)

# Log that the logger was configured
logger = logging.getLogger(__name__)
logger.info('Completed configuring logger()!')