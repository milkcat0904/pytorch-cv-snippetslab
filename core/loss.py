import torch.nn as nn

import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level = "INFO", format = FORMAT, datefmt = "[%X]", handlers = [RichHandler()])
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)

class Loss_Function():
    def __init__(self, hyper):
        self.criterion = nn.L1Loss()
        self.loss = {}

    def baseline_loss(self, output, meta, train):
        self.loss['loss'] = self.criterion(output, meta)
        return self.loss