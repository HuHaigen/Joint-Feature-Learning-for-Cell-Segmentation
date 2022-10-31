import argparse
import os
import os.path as osp
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.cell_dataset import CellDataset
from model.builder import build_model
from utils import util
from utils.config import Config
from utils.dice import *
from utils.logger import get_root_logger
from utils.path import mkdir_or_exist

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

ten2img = transforms.ToPILImage()


def predict(model, loader, device):
    """
    Return prediction masks by applying the model on the given dataset

    config:
        model (Unet3D): trained 3D UNet model used for prediction
        dataset (torch.utils.data.Dataset): input dataset
        out_channels (int): number of channels in the network output
        device (torch.Device): device to run the prediction on

    Returns:
         probability_maps (numpy array): prediction masks for given dataset
    """
    logger.info('Running prediction on {} patches...'.format(
        len(loader['val'])))
    accuracy_criterion = DiceAccuracy(threshold=0.5)

    val_accuracy = util.RunningAcc()
    total_pixel = 0
    total_P = 0
    total_T = 0
    total_F = 0
    total_TN = 0
    total_N = 0
    model.eval()
    with torch.no_grad():
        sample_num = 0
        for t in (loader['val']):
            data, target, corner, _ = t
            data, target = data.to(device), target.to(device)
            probs = model(data)

            sample_num += 1

            PT, A, B = accuracy_criterion(
                probs.detach().cpu().numpy(), target.detach().cpu().numpy())
            val_accuracy.update(PT, A, B)

            target = target.cpu().numpy()
            probs = probs.cpu().numpy()
            data = data.cpu()
            threshold = 0.5
            if probs.shape[1] != 1:
                temp1 = probs[:, 1, :, :]
            elif probs.shape[1] == 1:
                temp1 = probs

            _save_pic(data, temp1, target, sample_num)

            temp1[temp1 >= threshold] = 1
            temp1[temp1 < threshold] = 0
            fore_ground = temp1

            total_pixel += (fore_ground * target).sum()
            total_P += fore_ground.sum()
            total_T += target.sum()
            total_N += (1 - target).sum()
            total_F += (fore_ground * (1 - target)).sum()
            total_TN += ((1 - fore_ground) * (1 - target)).sum()

        logger.info('Accuracy: {}'.format(val_accuracy.avg))
        logger.info("ioU: {}".format(
            total_pixel / (total_P + total_T - total_pixel)))

        precession = total_pixel / total_P
        logger.info("precession:{}".format(precession))

        sensitivity = total_pixel / total_T
        logger.info("sensitivity:{}".format(sensitivity))


def _save_pic(data, probs, target, sample_num):
    path = cfg['predict_path']
    for folder in (['picture']):
        for data_kind in (['data', 'probs', 'target', 'contrast']):
            if os.path.exists(path + "/{}/{}".format(folder, data_kind)) == 0:
                os.makedirs(path + "/{}/{}".format(folder, data_kind))

    pic = Image.fromarray(probs[0] * 255).convert("L")
    pic.save(path + "/picture/probs/{}.bmp".format(sample_num))

    pic = Image.fromarray(target[0].astype(np.int8) * 255).convert("L")
    pic.save(path + "/picture/target/{}.bmp".format(sample_num))
    data = data[0]
    img = ten2img(data)
    img.save(path + "/picture/data/{}.jpg".format(sample_num))

    data_label = Image.open(path + "/picture/target/{}.bmp".format(sample_num))
    data_output = Image.open(path + "/picture/probs/{}.bmp".format(sample_num))
    if os.path.exists(path + "/picture/contrast/") == 0:
        os.makedirs(path + "/picture/contrast/")

    B = np.zeros((probs.shape[1], probs.shape[2]))
    B = np.uint8(B)
    B = Image.fromarray(B)

    img = Image.merge("RGB", (data_label, data_output, B))
    img.save(path + "/picture/contrast/{}.png".format(sample_num))


def _get_test_loaders():
    root = cfg['dataset_path']
    val_dataset = CellDataset(root, test_transforms, train=False)
    return {
        'val': DataLoader(val_dataset, batch_size=1, shuffle=False)
    }


def main():
    model = build_model(cfg)

    logger.info(model)

    logger.info('Loading model from {}...'.format(cfg['checkpoint']))
    util.load_checkpoint(cfg['checkpoint'], model)

    logger.info('Loading datasets...')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    loader = _get_test_loaders()

    predict(model, loader, device)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict the cell segmentation')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint path')
    parser.add_argument(
        '--predict-path', help='the dir to save predicting results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg['checkpoint'] = args.checkpoint

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.predict_path:
        cfg['predict_path'] = args.predict_path
    elif cfg['work_dir'] is not None:
        cfg['predict_path'] = osp.join(cfg.work_dir, 'result')
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg['predict_path'] = osp.join('./work_dirs',
                                       osp.splitext(osp.basename(args.config))[0], 'result')

    # predict_path
    mkdir_or_exist(osp.abspath(cfg.predict_path))

    # init the logger before other steps
    timestamp = time.strftime(r'%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.predict_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info(f'Config:\n{cfg.pretty_text}')

    main()
