import torch
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image
from torch.utils import data
from torchvision import transforms
from scipy.ndimage import gaussian_filter

class BaselineSet(data.Dataset):

    def __init__(self, imgs, smooth=False):
        super().__init__()
        self.smooth = smooth
        self.imgs = [x[:-10] for x in imgs]

    def __getitem__(self, idx):
        im = self.imgs[idx]
        target = Image.open('{}.seeds.png'.format(im))
        orig = Image.open('{}.plain.png'.format(im))
        return self.transform(orig, target)

    def transform(self, image, target):
        resize = transforms.Resize(900)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        image = resize(image)
        target = resize(target)
        target = ((np.array(target) > 0) * 255).astype('uint8')
        if self.smooth:
            target = gaussian_filter(target, sigma=2)
        target = Image.fromarray(target)
        return normalize(tf.to_tensor(image.convert('RGB'))), tf.to_tensor(target)

    def __len__(self):
        return len(self.imgs)
