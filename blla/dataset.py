import json
import torch
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image, ImageDraw
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
        with open('{}.path'.format(im), 'r') as fp:
            target = json.load(fp)
        orig = Image.open('{}.plain.png'.format(im))
        return self.transform(orig, target)

    def transform(self, image, target):
        resize = transforms.Resize(1200)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        orig_size = image.size
        image = resize(image)
        scale = image.size[0]/orig_size[0]
        t = Image.new('L', image.size)
        draw = ImageDraw.Draw(t)
        lines = []
        for line in target:
            l = []
            for point in line:
                l.append((int(point[0]*scale), int(point[1]*scale)))
            draw.line(l, fill=255, width=10)
        del draw
        target = np.array(t)
        if self.smooth:
            target = gaussian_filter(target, sigma=2)
        target = Image.fromarray(target)
        return normalize(tf.to_tensor(image.convert('RGB'))), tf.to_tensor(target)

    def __len__(self):
        return len(self.imgs)
