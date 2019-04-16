import json
import torch
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image
from torch.utils import data
from torchvision import transforms

from scipy.ndimage import morphology

class InitialVertexDataset(data.Dataset):

    def __init__(self, imgs):
        super().__init__()
        self.imgs = [x[:-10] for x in imgs]

    def __getitem__(self, idx):
        im = self.imgs[idx]
        with open('{}.lines.json'.format(im)) as fp:
            target = json.load(fp)
        feats = torch.load('{}.feat'.format(im))
        return self.transform(feats, target)

    def transform(self, feats, targets):
        vertices = [min(t, key=lambda x: x[0]) for t in target['lines']]
        scale_x = feats.shape[3] / target['orig'][0]
        scale_y = feats.shape[2] / target['orig'][1]
        tim = np.zeros(image.size)
        for v in vertices:
            tim[int(v[0]*scale_x), int(v[1]*scale_y)] = 255
        tim = morphology.binary_dilation(tim, iterations=2)*255
        target = Image.fromarray(tim.T.astype('uint8'))
        return feats, tf.to_tensor(target)

    def __len__(self):
        return len(self.imgs)

class VerticesDataset(data.Dataset):

    def __init__(self, imgs):
        super().__init__()
        self.imgs = [x[:-10] for x in imgs]
        self.len = 0
        for im in self.imgs:
            with open('{}.lines.json'.format(im)) as fp:
                self.len += len(json.load(fp))

    def __getitem__(self, idx):
        im = self.imgs[idx]
        with open('{}.lines.json'.format(im)) as fp:
            target = json.load(fp)
        feats = torch.load('{}.feat'.format(im))
        # create time series of targets
        l = np.random.choice(target['lines'])
        # TODO: scale targets
        if l[0][0] > l[-1][0]:
            l = list(reversed(l))
        points = [line(*start, *end) for start, end in zip(l, l[1::])]
        x = np.hstack([x[0] for x in points])[::10]
        y = np.hstack([x[1] for x in points])[::10]
        targets = []
        for target in np.array((x, y)).T:
            t = np.zeros(target['orig'])
            t[target.T] = 1
            targets.append(t)
        return feats, torch.tensor(np.stack(targets))

    def __len__(self):
        return len(self.imgs)
