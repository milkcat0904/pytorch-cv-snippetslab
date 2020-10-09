import os
import time
import torch
import wandb
import yaml
import tools.monitor as utils

from core.loss import Loss_Function

import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(level='INFO', format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)
from rich.traceback import install
install()

class Train():
    def __init__(self, cfgs, model, trainloader, valloader):
        self.cfgs = cfgs
        self.para_cfgs = yaml.load(open(self.cfgs['model']['hyperPara_path'], 'r'), Loader = yaml.FullLoader)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.trainLoader = trainloader
        self.valLoader = valloader
        self.model = model
        self.save_folder = os.path.join(self.para_cfgs['train']['save_folder'],
                                        self.cfgs['wandb']['name'])
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        self.val_loss = 999

        self.lr = self.para_cfgs['train']['lr']
        self.weight_decay = self.para_cfgs['train']['weight_decay']
        self.momentum = self.para_cfgs['train']['momentum']
        self.step_size = self.para_cfgs['train']['step_size']
        self.gamma = self.para_cfgs['train']['gamma']
        self.epochs = self.para_cfgs['train']['epochs']
        self.save_freq = self.para_cfgs['train']['save_freq']
        self.milestones = self.para_cfgs['train']['milestones']

        self.print_freq = self.para_cfgs['train']['print_freq']
        self.use_wandb = self.cfgs['wandb']['use']
        self.monitors_list = self.cfgs['wandb']['monitors']

        # loss function
        self.loss = Loss_Function(self.para_cfgs)
        self.loss_type = self.para_cfgs['model']['loss_type']

        self.optimizer = torch.optim.Adam(params = self.params, lr = self.lr,
                                          weight_decay = self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.milestones,
                                                                 gamma = self.gamma)

        # metric
        self.metric_dic = {}
        self.start_train()

    def start_train(self):
        for epoch in range(self.epochs):

            # train 1 epoch
            self.train_one_epoch(epoch)
            self.lr_scheduler.step()

            self.evaluate()

            if self.use_wandb:
                wandb.log(self.metric_dic)

            if self.metric_dic['eval_loss'] < self.val_loss or epoch%self.save_freq == 0:
                self.save_model(epoch, self.metric_dic['eval_loss'])
                self.val_loss = self.metric_dic['eval_loss']

    def save_model(self, epoch, loss):

        if self.save_folder:
            checkpoint = {
                'loss': loss,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': epoch}

            utils.save_on_master(
                checkpoint,
                os.path.join(self.save_folder, 'model_{0}_lr{1}.pth'.format(self.cfgs['wandb']['name'], self.lr)))

            if epoch%self.save_freq == 0:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(self.save_folder, 'checkpoint_{0}_lr{1}_e{2}.pth'.format(self.cfgs['wandb']['name'],
                                                                                          self.lr, epoch)))
            logger.info('Saving model in {} folder.'.format(self.save_folder))

    def backprop(self, output, meta, start_time, metric_logger, train=True):

        loss_dict = eval('self.loss.'+self.loss_type)(output, meta, train)
        if train == True:
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()

        metric_logger.update(loss = loss_dict['loss'].item(),
                             lr = self.optimizer.param_groups[0]["lr"])

        if type(output) == tuple:
            batch_size = output[0].shape[0]
        else:
            batch_size = output.shape[0]
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    def train_one_epoch(self, epoch):
        # types: image -> joint2d, joint3d

        self.model.train()
        # 进度条初始化
        metric_logger = utils.MetricLogger(delimiter = "  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size = 1, fmt = '{value}'))
        metric_logger.add_meter('img/s', utils.SmoothedValue(window_size = 10, fmt = '{value:.1f}'))
        header = 'Epoch: [{}]'.format(epoch)

        # iteration
        for image,  metadata in metric_logger.log_every(self.trainLoader, self.print_freq, header):
            start_time = time.time()

            image = image.to('cuda')
            output = self.model(image) # (64, 105)
            self.backprop(output, metadata, start_time, metric_logger) ###

        for i in self.monitors_list:
            self.metric_dic['train_' + i] = metric_logger.meters[i].avg

    def evaluate(self):

        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter = "  ")
        header = 'Test:'

        with torch.no_grad():
            for image, metadata in metric_logger.log_every(self.valLoader, self.print_freq, header):
                start_time = time.time()
                image = image.to('cuda')
                output = self.model(image)  # (64, 105)
                self.backprop(output, metadata, start_time, metric_logger, train = False)

        for i in self.monitors_list:
            self.metric_dic['eval_' + i] = metric_logger.meters[i].avg