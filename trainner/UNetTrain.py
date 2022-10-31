import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tensorboardX import SummaryWriter
from utils import dice, util
from utils.logger import get_root_logger


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        pred = torch.cat((1-pred, pred), dim=1)
        target = torch.cat((1-target, target), dim=1)
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        log_p = probs.log()

        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1-self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class UNetTrainer:

    def __init__(self, model, optimizer, loss_criterion,
                 use_density,
                 accuracy_criterion,
                 device, loaders, work_dir,
                 max_num_epochs=200, max_num_iterations=1e5, max_patience=20,
                 validate_after_iters=100, log_after_iters=100,
                 best_val_accuracy=float('-inf'),
                 num_iterations=0, num_epoch=0, pred_result="", use_corner=0, logger=None):
        if logger is None:
            self.logger = get_root_logger()
        else:
            self.logger = logger

        self.logger.info("Sending the model to '{}'".format(device))
        self.model = model.to(device)
        self.logger.debug(model)

        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.accuracy_criterion = accuracy_criterion
        self.device = device
        self.loaders = loaders
        self.work_dir = work_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.pred_result = pred_result
        self.use_corner = use_corner
        self.best_val_accuracy = best_val_accuracy
        self.writer = SummaryWriter(
            log_dir=os.path.join(work_dir, 'logs'))
        self.edge_loss = FocalLoss()

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        # used for early stopping
        self.max_patience = max_patience
        self.patience = max_patience
        self.dice_criterion = dice.DiceAccuracy(threshold=0.5)
        self.use_density = use_density
        self.density_loss = torch.nn.MSELoss()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, loss_criterion,
                        density_loss,
                        accuracy_criterion, loaders,
                        validate_after_iters,
                        log_after_iters,
                        pred_result="",
                        use_corner=False,
                        logger=None):
        logger.info("Loading checkpoint '{}'...".format(checkpoint_path))
        
        state = util.load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            "Checkpoint loaded. Epoch: {}. Best val accuracy: {:.5f}. Num_iterations: {}".format(
                state['epoch'], state['best_val_accuracy'], state['num_iterations']
            ))
        work_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, loss_criterion,
                   density_loss,
                   accuracy_criterion, torch.device(state['device']), loaders,
                   work_dir,
                   best_val_accuracy=state['best_val_accuracy'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   max_patience=state['max_patience'],
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   pred_result=pred_result,
                   use_corner=use_corner,
                   logger=logger)

    def train(self):
        for epoch in range(self.max_num_epochs):
            should_terminate = self.train_one_epoch(self.loaders['train'])

            if should_terminate:
                break

            self.num_epoch = epoch

    def train_one_epoch(self, train_loader):
        train_losses = util.RunningAverage()
        self.model.train()
        PT, A, B = 0, 0, 0
        for i, t in enumerate(train_loader):
            input, target, edge, density = t
            input, target, edge = input.to(self.device), target.to(
                self.device), edge.to(self.device)
            density = density.to(self.device)
            output, loss = self._forward_pass(input, target, edge, density)
            output = self.model.final_activation(output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.num_iterations += 1
            train_losses.update(loss.item(), input.size(0))

            pt, a, b = self.dice_criterion(
                output.detach().cpu().numpy(), target.detach().cpu().numpy())
            PT += pt
            A += a
            B += b

            if (i + 1) % self.log_after_iters == 0 or i + 1 == len(train_loader):
                self.logger.info(
                    f'Epoch [{self.num_epoch}/{self.max_num_epochs}]: Training iteration [{i + 1}/{len(train_loader)}], lr {self._get_lr()}'
                )

        self.logger.info(
            'Training stats. Total_loss: {:.8f}, Dice: {}'.format(train_losses.avg, 2*PT/(A+B+1e-8)))

        train_stats = dict(
            train_loss_avg=train_losses.avg,
            train_lr=self._get_lr()
        )

        self._log_stats(train_stats, self.num_iterations)
        dice, iou, precession, sensitivity = self.validate(self.loaders['val'])

        val_stats = dict(
            Dice=dice,
            IoU=iou,
            Precession=precession,
            Sensitivity=sensitivity
        )

        self._log_stats(val_stats, self.num_epoch)

        is_best = self._is_best_val_accuracy(dice)
        self._save_checkpoint(is_best)

        if self._check_early_stopping(is_best):
            self.logger.info(
                'Validation accuracy did not improve for the last {} validation runs. Early stopping...'.format(
                    self.max_patience
                ))
            return True

        if self.max_num_iterations < self.num_iterations:
            self.logger.info(
                'Maximum number of iterations {} exceeded. Finishing training...'.format(
                    self.max_num_iterations
                ))
            return True

        return False

    def validate(self, val_loader):
        self.logger.info('epoch: {}, Validating...'.format(self.num_epoch))
        PT, A, B = 0, 0, 0
        total_pixel, total_P, total_T, total_F, total_TN, total_N = 0, 0, 0, 0, 0, 0
        val_losses = util.RunningAverage()
        save_path = self.pred_result
        if self.num_epoch % 10 == 0:
            if not os.path.exists(save_path + "/{}".format(self.num_epoch)):
                os.makedirs(save_path + "/{}".format(self.num_epoch))
        self.model.eval()
        try:
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # self.logger.info('Validation iteration {}'.format(i))
                    input, target, edge, density = data
                    input, target, edge = input.to(self.device), target.to(
                        self.device), edge.to(self.device)
                    density = density.to(self.device)
                    output, loss = self._forward_pass(
                        input, target, edge, density)
                    val_losses.update(loss.item(), input.size(0))
                    if self.num_epoch % 10 == 0:
                        tmp = output.detach().cpu().numpy()[0, 1, :, :]
                        tmp = (tmp - tmp.min()) / tmp.ptp()
                        a = np.matrix(tmp * 255)
                        R = Image.fromarray(np.matrix(a)).convert("L")
                        R.save(save_path + "/{}/{}.bmp".format(self.num_epoch, i))

                    pt, a, b = self.dice_criterion(
                        output.detach().cpu().numpy(), target.detach().cpu().numpy())
                    PT += pt
                    A += a
                    B += b
                    target = target.cpu().numpy()
                    probs = output.cpu().numpy()
                    threshold = 0.5
                    if probs.shape[1] != 1:
                        temp1 = probs[:, 1, :, :]
                    elif probs.shape[1] == 1:
                        temp1 = probs

                    temp1[temp1 >= threshold] = 1
                    temp1[temp1 < threshold] = 0
                    fore_ground = temp1

                    total_pixel += (fore_ground * target).sum()
                    total_P += fore_ground.sum()
                    total_T += target.sum()
                    total_N += (1 - target).sum()
                    total_F += (fore_ground * (1 - target)).sum()
                    total_TN += ((1 - fore_ground) * (1 - target)).sum()

                dice = 2*PT/(A+B+1e-8)
                iou = total_pixel / (total_P + total_T - total_pixel)

                self.logger.info(
                    'Validation finished. Loss: {}.'.format(val_losses.avg))
                self.logger.info('Dice: {}'.format(dice))
                self.logger.info("IoU: {}".format(iou))

                precession = total_pixel / total_P
                self.logger.info("Precession: {}".format(precession))

                sensitivity = total_pixel / total_T
                self.logger.info("Sensitivity: {}".format(sensitivity))
                self.logger.info('BestAccuracy: {}'.format(
                    self.best_val_accuracy))
        finally:
            self.model.train()
        return dice, iou, precession, sensitivity

    def _forward_pass(self, input, target, corner, density):
        output = self.model(input)

        sum_loss = self.loss_criterion(output, target)
        if self.use_corner != 0:
            sum_loss += self.edge_loss(output, corner) * self.use_corner
        if self.use_density != 0:
            sum_loss += self.density_loss(output, density) * self.use_density
        return output, sum_loss

    def _check_early_stopping(self, best_model_found):
        if best_model_found:
            self.patience = self.max_patience
        else:
            self.patience -= 1
            if self.patience <= 0:
                # early stop the training
                return True
            # adjust learning rate when reaching half of the max_patience
            if self.patience == self.max_patience // 2:
                self._adjust_learning_rate()
        return False

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def _adjust_learning_rate(self, decay_rate=0.1):
        """Sets the learning rate to the initial LR decayed by 'decay_rate'"""

        old_lr = self._get_lr()
        assert old_lr > 0, 'Previous lr is not more than 0.'
        new_lr = decay_rate * old_lr
        self.logger.info(
            'Changing learning rate from {} to {}'.format(old_lr, new_lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _is_best_val_accuracy(self, val_accuracy):
        is_best = val_accuracy > self.best_val_accuracy
        if is_best:
            self.logger.info(
                'Saving new best validation accuracy: {}'.format(val_accuracy))
        self.best_val_accuracy = max(val_accuracy, self.best_val_accuracy)
        return is_best

    def _save_checkpoint(self, is_best):
        util.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'max_patience': self.max_patience
        }, is_best, work_dir=self.work_dir,
            logger=self.logger)

    def _log_stats(self, train_stats: dict, step):
        for tag, value in train_stats.items():
            self.writer.add_scalar(tag, value, step)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(),
                                      self.num_iterations)
            self.writer.add_histogram(name + '/grad',
                                      value.grad.data.cpu().numpy(),
                                      self.num_iterations)
