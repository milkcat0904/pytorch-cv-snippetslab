import os
import json
import torch
import wandb

from rich.progress import track
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")

from rich.traceback import install
install()
import torch.utils.data as data

from prefetch_generator import BackgroundGenerator

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Dataset(data.Dataset):
    def __init__(self, cfg, hyper, mode = 'train'):
        self.mode = mode
        self.cfgs = cfg
        self.para_cfgs = hyper
        self.dataset_path = cfg['dataset']['dataset_folder']
        self.label_folder = cfg['dataset']['label_folder']
        self.gt_index = cfg['dataset']['gt_index']

        self.imageName = []
        self.dataDict = {}

        # wandb
        self.use_wandb = self.cfgs['wandb']['use']
        if self.use_wandb:
            self.wandb_config()

        if mode == 'train':

            self.train_json_path_list = cfg['dataset']['train_json']
            self.train_folder_list = cfg['dataset']['train_folder']
            logger.info('Training dataset folder: {}'.format(self.train_folder_list))

            self.brightness = cfg['augmentation']['brightness']
            self.contrast = cfg['augmentation']['contrast']

            self.erasing_prop = cfg['augmentation']['prop_thresh']['erasing']

            self.transform = transforms.Compose([transforms.ColorJitter(brightness = self.brightness,
                                                                        contrast = self.contrast),
                                                 transforms.Resize((80, 80)),
                                                 transforms.ToTensor()
                                                 ])

            for index in range(len(self.train_folder_list)):
                tmpJsonPath = os.path.join(self.label_folder, self.train_json_path_list[index])
                self.phrase_json(self.train_folder_list[index], tmpJsonPath)

        else:
            self.test_folder_list = cfg['dataset']['test_folder']
            self.test_json_path_list = cfg['dataset']['test_json']
            logger.info('Test dataset folder: {}'.format(self.test_folder_list))

            self.transform = transforms.Compose([transforms.Resize((80, 80)),
                                                 transforms.ToTensor()])
            self.use_custom = False
            logger.info('Data Augmentation: {}'.format('None'))

            for index in range(len(self.test_folder_list)):
                tmpJsonPath = os.path.join(self.label_folder, self.test_json_path_list[index])
                self.phrase_json(self.test_folder_list[index], tmpJsonPath)

        # training data&label
        self.gt = []
        # self.make_dataset()
        logger.info('Data file number: {}'.format(len(self.imageName)))
        self.make_dataset()
        self.gt = np.array(self.gt)

    def wandb_config(self):
        note = ''

        hyperparameter_defaults = dict()

        wandb.init(project = self.cfgs['wandb']['project_name'],
                   config = hyperparameter_defaults,
                   name = self.cfgs['wandb']['name'],
                   tags = self.cfgs['wandb']['tags'],
                   notes = note)

    def phrase_json(self, image_folder, json_path):
        data = os.path.join(self.dataset_path, image_folder)

        logger.info('Read json file: {} ...'.format(json_path))
        dataDict = json.load(open(json_path))
        for tmpName in track(dataDict.keys()):
            self.imageName.append(os.path.join(data, tmpName + '.png'))
        self.dataDict.update(dataDict)

    def make_dataset(self):

        logger.info('Start making {} dataloader...'.format(self.mode))

        for i in track(range(len(self.imageName))):
            fileName = os.path.split(self.imageName[i])[-1].split('.')[0]
            tmpGT = self.dataDict[fileName]['ground_truth']
            self.gt.append(tmpGT) # list(map(float, tmpGT.split(' ')[:-1]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: image, addon, metadata
        """
        imagePath = self.imageName[index]
        image = Image.open(imagePath)
        gt = self.gt[index]

        metadata = {}

        # No Transform
        if self.transform is not None:
            image = self.transform(image)

        return image, gt

    def __len__(self):
        return len(self.imageName)