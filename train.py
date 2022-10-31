import argparse
import os
import os.path as osp
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data.cell_dataset import CellDataset
from model.builder import build_model
from trainner.UNetTrain import UNetTrainer
from utils.config import Config
from utils.dice import *
from utils.logger import get_root_logger
from utils.path import mkdir_or_exist

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def _create_optimizer(config, model):
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _get_loaders():
    root = cfg['dataset_path']
    train_dataset = CellDataset(
        root, train_transforms, train=True, use_density=cfg['use_density'])
    val_dataset = CellDataset(root, test_transforms,
                              train=False, use_density=cfg['use_density'])
    return {
        'train': DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True),
        'val': DataLoader(val_dataset, batch_size=1, shuffle=False)
    }





def main():
    path = cfg['dataset_path']
    if cfg['resample'] is True:
        for picture in ['image', 'label', 'corner', 'density']:
            dst = path + '/val/{}/'.format(picture)
            src = path + '/train/{}/'.format(picture)
            if not os.path.exists(dst):
                os.makedirs(dst)
            files = os.listdir(dst)
            for file in files:
                shutil.move(dst + file, src + file)
        if cfg['resample_by_manual'] is not False:
            files = cfg['resample_by_manual']
        else:
            nums = len(os.listdir(path + "/train/label"))
            files = np.random.choice(
                range(1, nums), cfg['resample_num'], replace=False)

        logger.info(files)
        for x in ['image', 'label', 'corner', 'density']:
            dst = path + '/val/{}/'.format(x)
            src = path + '/train/{}/'.format(x)
            suffix = '.bmp' if x == 'label' else '.jpg'
            for file in files:
                shutil.move(src + str(file) + suffix, dst + str(file) + suffix)

        logger.info("Resample val dataset, ID:")
        logger.info(files)

    # Get device to train on
    torch.manual_seed(cfg['randseed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    logger.info(device)

    loss_criterion = nn.CrossEntropyLoss()

    accuracy_criterion = DiceAccuracy()

    model = build_model(cfg).to(device)
    logger.info(model)

    loaders = _get_loaders()

    # Create the optimizer
    optimizer = _create_optimizer(cfg, model)
    num_iters = len(loaders['train'])

    if cfg['resume']:
        logger.info("From the last saved model: {}".format(cfg['resume']))
        trainer = UNetTrainer.from_checkpoint(cfg['resume'], model, optimizer,
                                              loss_criterion,
                                              cfg['use_density'],
                                              accuracy_criterion,
                                              loaders,
                                              validate_after_iters=num_iters,
                                              log_after_iters=num_iters,
                                              pred_result=cfg['pred_result'],
                                              use_corner=cfg['use_corner'],
                                              logger=logger)
    else:
        logger.info("Start a new training...")
        trainer = UNetTrainer(model, optimizer,
                              loss_criterion,
                              cfg['use_density'],
                              accuracy_criterion,
                              device, loaders, cfg['work_dir'],
                              max_num_epochs=cfg['epochs'],
                              max_num_iterations=cfg['iters'],
                              max_patience=cfg['patience'],
                              validate_after_iters=num_iters,
                              log_after_iters=num_iters // 5,
                              pred_result=cfg['pred_result'],
                              use_corner=cfg['use_corner'],
                              logger=logger)
    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser(description='Train the cell segmentation')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg['work_dir'] = osp.join('./work_dirs',
                                   osp.splitext(osp.basename(args.config))[0])

    cfg['pred_result'] = osp.join(cfg.work_dir, 'pred_results')

    # create work_dir & pred_result
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    mkdir_or_exist(osp.abspath(cfg.pred_result))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime(r'%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info(f'Config:\n{cfg.pretty_text}')

    main()
