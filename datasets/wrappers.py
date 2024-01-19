import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from .resample_transform import Resampler
from torchvision.transforms import InterpolationMode


def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False, interpolation_mode="nearest", resampling_factor = 1, resampling_inp_size = None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment


        self.img_transform = transforms.Compose([
                Resampler(inp_size = resampling_inp_size, interpolation_mode=interpolation_mode, resampling_factor=resampling_factor),
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                Resampler(inp_size = resampling_inp_size, interpolation_mode=interpolation_mode, resampling_factor=resampling_factor),
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }


@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, interpolation_mode="nearest", resampling_factor = 1, resampling_inp_size = None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
                Resampler(inp_size = resampling_inp_size, interpolation_mode=interpolation_mode, resampling_factor=resampling_factor),
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                Resampler(inp_size = resampling_inp_size, interpolation_mode=interpolation_mode, resampling_factor=resampling_factor),
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # random filp
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        mask =      (mask)

        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }
    


@register('infer')
class InferDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False, interpolation_mode="nearest", resampling_factor = 1, resampling_inp_size = None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment


        self.img_transform = transforms.Compose([
                Resampler(inp_size = resampling_inp_size, interpolation_mode=interpolation_mode, resampling_factor=resampling_factor),
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                Resampler(inp_size = resampling_inp_size, interpolation_mode=interpolation_mode, resampling_factor=resampling_factor),
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]

        return {
            'inp': self.img_transform(img)
        }
