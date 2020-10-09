import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

import yaml
import torch.utils.data
from core.dataset import  Dataset, DataLoaderX
from models.select_models import select_model
from core.trainer import Train
import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)

from rich.traceback import install
install()

if __name__ == '__main__':

    cfg_path = 'config/train.yaml'
    cfgs = yaml.load(open(cfg_path), Loader = yaml.FullLoader)
    hyper_para = yaml.load(open(cfgs['model']['hyperPara_path'], 'r'), Loader = yaml.FullLoader)

# =================================================================================================== #
#                                           1. Dataloader                                             #
# =================================================================================================== #

    trainset = Dataset(cfgs, hyper_para)
    trainloader = DataLoaderX(trainset, batch_size = hyper_para['train']['batch_size'],
                              shuffle = hyper_para['train']['shuffle'],
                              num_workers = hyper_para['train']['num_worker'],
                              pin_memory = True)

    valset = Dataset(cfgs, hyper_para, mode = 'val')
    valloader = DataLoaderX(valset, batch_size = cfgs['val']['batch_size'], shuffle = cfgs['val']['shuffle'],
                            num_workers = cfgs['val']['num_worker'])

# =================================================================================================== #
#                                              2. Model                                               #
# =================================================================================================== #

    logger.info('Model init & loading pretrain model...')
    model = select_model(cfgs, hyper_para)
    model = torch.nn.DataParallel(model)

# =================================================================================================== #
#                                              3. Train                                               #
# =================================================================================================== #

    logger.info('Start training...')
    Train(cfgs, model, trainloader, valloader)