from PIL import Image
import numpy as np


class DiceAccuracy:
    def __init__(self, threshold=0.5, epsilon=1e-8):
        self.epsilon = epsilon
        self.threshold = threshold

    def __call__(self, output, target):
        if output.shape[1] != 1:
            temp = output[:, 1, :, :].copy()
        else:
            temp = output.copy()

        temp[temp >= self.threshold] = 1
        temp[temp < self.threshold] = 0
        fore_ground = temp
        
        PT = (fore_ground * target).sum()
        A = fore_ground.sum()
        B = target.sum()
        return PT, A, B
