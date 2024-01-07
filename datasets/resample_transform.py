import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image
import os
from PIL import Image

class Resampler(torch.nn.Module):

    def __init__(self, inp_size, interpolation_mode, resampling_factor):
        super(Resampler, self).__init__()
        self.inp_size = inp_size
        self.resampling_factor = resampling_factor
        self.interpolation_mode = InterpolationMode(interpolation_mode)
    
    def forward(self, img):
        new_size = int(self.inp_size // self.resampling_factor)

        downsampler = transforms.Resize(size=(new_size,new_size), interpolation=self.interpolation_mode)
        upsampler = transforms.Resize(size=(self.inp_size, self.inp_size), interpolation=self.interpolation_mode)
        
        downsampled_image = downsampler.forward(img)
        transformed_image = upsampler.forward(downsampled_image)

        return transformed_image