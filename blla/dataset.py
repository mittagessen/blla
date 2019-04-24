import json
import torch
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image
from torch.utils import data
from torchvision import transforms

from scipy.ndimage import morphology
from skimage.draw import line

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

    def transform(self, feats, target):
        vertices = [min(t, key=lambda x: x[0]) for t in target['lines']]
        scale_x = feats['ds_2'].shape[2] / (target['orig'][1])
        scale_y = feats['ds_2'].shape[1] / (target['orig'][0])
        tim = np.zeros(feats['ds_2'].shape[1::])
        for v in vertices:
            tim[int(v[1]*scale_y), int(v[0]*scale_x)] = 255
        tim = morphology.binary_dilation(tim, iterations=2)*255
        target = Image.fromarray(tim.astype('uint8'))
        return (feats['ds_2'],
                feats['ds_3'],
                feats['ds_4'],
                feats['ds_5'],
                feats['ds_6'],
                tf.to_tensor(target).squeeze())

    def __len__(self):
        return len(self.imgs)

class VerticesDataset(data.Dataset):

    def __init__(self, imgs):
        super().__init__()
        self.imgs = [x[:-10] for x in imgs]
        self.len = 0
        for im in self.imgs:
            with open('{}.lines.json'.format(im)) as fp:
                self.len += len(json.load(fp)['lines'])

    def __getitem__(self, idx):
        im = self.imgs[idx]
        with open('{}.lines.json'.format(im)) as fp:
            target = json.load(fp)
        feats = torch.load('{}.feat'.format(im))
        fshape = feats['ds_2'].shape[1:]
        # feature shapes + 1 for EOS symbol
        tshape = (fshape[0], fshape[1] + 1)
        # scale factors between original image and downsampled feature map
        scale_y = fshape[0] / (target['orig'][0])
        scale_x = fshape[1] / (target['orig'][1])
        # create time series of targets
        l = target['lines'][np.random.choice(len(target['lines']))]
        if l[0][0] > l[-1][0]:
            l = list(reversed(l))
        points = [line(*start, *end) for start, end in zip(l, l[1::])]
        y = np.hstack([x[1]*scale_y for x in points])[::10].astype('int')
        x = np.hstack([x[0]*scale_x for x in points])[::10].astype('int')
        # create one target map for each time step
        targets = []
        for ty, tx in zip(y, x):
            t = np.zeros(tshape)
            t[ty, tx] = 255
            targets.append(t)
            #targets.append(morphology.binary_dilation(t, iterations=2)*255)
        # add initial node twice
        targets.insert(1, targets[0])
        # zip into inputs of shape (seq, stack, H, W)
        inputs = torch.tensor([np.array(x) for x in zip(targets, targets[1::])])
        # add EOS symbol
        t = np.zeros(tshape)
        t[0, -1] = 255
        targets.append(t)
        # build targets (seq, H, W)
        targets = torch.tensor(targets[2::])
        return (feats['ds_2'],
                feats['ds_3'],
                feats['ds_4'],
                feats['ds_5'],
                feats['ds_6'],
                inputs.float(),
                targets.float())

    def __len__(self):
        return self.len
