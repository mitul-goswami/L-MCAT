import torch
import random
import numpy as np

class TransformCompose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, sar, optical):
        for t in self.transforms:
            sar, optical = t(sar, optical)
        return sar, optical

class RandomRotate:
    def __init__(self):
        self.angles = [0, 90, 180, 270]
        
    def __call__(self, sar, optical):
        angle = random.choice(self.angles)
        sar = self._rotate(sar, angle)
        optical = self._rotate(optical, angle)
        return sar, optical
    
    def _rotate(self, img, angle):
        return torch.rot90(img, angle // 90, [1, 2])

class GaussianNoise:
    def __init__(self, sigma=0.02):
        self.sigma = sigma
        
    def __call__(self, sar, optical):
        noise = torch.randn_like(sar) * self.sigma
        return sar + noise, optical

class ColorJitter:
    def __init__(self, brightness=0.1):
        self.brightness = brightness
        
    def __call__(self, sar, optical):
        factor = 1 + (torch.rand(1) * 2 - 1) * self.brightness
        return sar, optical * factor
