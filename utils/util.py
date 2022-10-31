import logging
import os
import shutil
import sys

import numpy as np
import torch


def save_checkpoint(state, is_best, work_dir, logger=None):
    """Saves model and training parameters at '{work_dir}/last_checkpoint.pth'.
    If is_best==True saves '{work_dir}/best_checkpoint.pth' as well.

    Args:
        `state` (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best validation accuracy so far
        `is_best` (bool): if True state contains the best model seen so far
        `work_dir` (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(work_dir):
        log_info(
            "Checkpoint directory does not exists. Creating {}".format(work_dir))
        os.mkdir(work_dir)

    last_file_path = os.path.join(work_dir, 'last_checkpoint.pth')
    log_info("Saving last checkpoint")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(
            work_dir, 'best_checkpoint.pth')
        log_info("Saving best checkpoint")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model and training parameters from a given `checkpoint`
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        `checkpoint` (string): path to the checkpoint to be loaded
        `model` (torch.nn.Module): model into which the parameters are to be copied
        `optimizer` (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint):
        raise IOError("Checkpoint '{}' does not exist".format(checkpoint))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    state = torch.load(checkpoint, map_location="cuda:0")
    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


class RunningAcc:
    """Computes and stores the average
    """

    def __init__(self):
        self.pixel = 0
        self.P = 0
        self.T = 0
        self.avg = 0

    def update(self, pixel, P, T):
        self.pixel += pixel
        self.P += P
        self.T += T
        self.avg = 2 * self.pixel / (self.P + self.T + 1e-8)


class RunningHit:
    """Computes and stores the average
    """

    def __init__(self):
        self.num = 0
        self.hit = 0
        self.avg = 0

    def update(self, num, hit):
        self.num += num
        self.hit += hit
        self.avg = self.hit / (self.num + 1e-8)
