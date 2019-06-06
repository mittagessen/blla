#!/usr/bin/env python3

import os
import glob
import json
import click
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn, optim
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import DataLoader

from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, TerminateOnNan
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage, Loss
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator

from blla import model
from blla.dataset import BaselineSet
from blla.postprocess import denoising_hysteresis_thresh, vectorize_lines

@click.group()
def cli():
    pass

@cli.command()
@click.option('-n', '--name', default=None, help='prefix for checkpoint file names')
@click.option('-i', '--load', default=None, type=click.Path(exists=True, readable=True), help='pretrained weights to load')
@click.option('-l', '--lrate', default=2e-4, help='initial learning rate')
@click.option('--weight-decay', default=1e-5, help='weight decay')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('--smooth/--no-smooth', default=False, help='enables smoothing of the targets in the data sets.')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-v', '--validation', default='val', help='validation set location')
@click.option('-t', '--arch', default='RecLabelNet', type=click.Choice(['RecResUNet', 'ResUNet', 'RecLabelNet']))
@click.option('--loss', default='MSELoss', type=click.Choice(['MSELoss', 'BCELoss']))
@click.argument('ground_truth', nargs=1)
def train(name, load, lrate, weight_decay, workers, smooth, device, validation, arch, loss, ground_truth):

    if not name:
        name = '{}_{}'.format(lrate, weight_decay)
    click.echo('model output name: {}'.format(name))

    torch.set_num_threads(1)

    train_set = BaselineSet(glob.glob('{}/**/*.plain.png'.format(ground_truth), recursive=True), smooth=smooth)
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=1, shuffle=True, pin_memory=True)
    val_set = BaselineSet(glob.glob('{}/**/*.plain.png'.format(validation), recursive=True), smooth=False)
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=1, pin_memory=True)

    click.echo('loading network')
    net = getattr(model, arch)()

    if load:
        click.echo('loading weights')
        net = torch.load(load, map_location=device)

    criterion = getattr(nn, loss)()
    opti = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lrate, weight_decay=weight_decay)

    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    def output_preprocess(output):
        o, target = output
        o = denoising_hysteresis_thresh(o.detach().squeeze().cpu().numpy(), 0.4, 0.5, 0)
        return torch.from_numpy(o.astype('f')).unsqueeze(0).unsqueeze(0).to(device), target.double().to(device)

    trainer = create_supervised_trainer(net, opti, criterion, device=device, non_blocking=True)
    accuracy = Accuracy(output_transform=output_preprocess)
    precision = Precision(output_transform=output_preprocess)
    recall = Recall(output_transform=output_preprocess)
    loss = Loss(criterion)
    evaluator = create_supervised_evaluator(net, device=device, non_blocking=True)

    accuracy.attach(evaluator, 'accuracy')
    precision.attach(evaluator, 'precision')
    recall.attach(evaluator, 'recall')
    loss.attach(evaluator, 'loss')

    ckpt_handler = ModelCheckpoint('.', name, save_interval=1, n_saved=100, require_empty=False)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, ['loss'])

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=ckpt_handler, to_save={'net': net})
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=TerminateOnNan())

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_data_loader)
        metrics = evaluator.state.metrics
        r = metrics['recall']
        p = metrics['precision']
        f1 = (p  * r * 2)/(p + r + 1e-20)
        progress_bar.log_message('eval results - epoch {} loss: {:.4f} f1: {:.4f}, accuracy: {:.4f} recall: {:.4f} precision {:.4f}'.format(engine.state.epoch,
                                                                                                                   metrics['loss'],
                                                                                                                   f1,
                                                                                                                   metrics['accuracy'],
                                                                                                                   p,
                                                                                                                   r,
                                                                                                                   metrics['precision']))
    trainer.run(train_data_loader, max_epochs=200)

@cli.command()
@click.option('-m', '--model', default=None, help='model file')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-c', '--context', default=80, help='context around baseline')
@click.option('-t', '--thresholds', default=(0.3, 0.5), type=(float, float), help='thresholds for hysteresis thresholding')
@click.option('-s', '--sigma', default=2.5, help='sigma of gaussian filter in postprocessing')
@click.argument('images', nargs=-1)
def pred(model, device, context, thresholds, sigma, images):
    device = torch.device(device)
    with open(model, 'rb') as fp:
        m = torch.load(fp, map_location=device)

    resize = transforms.Resize(1200)
    transform = transforms.Compose([transforms.Resize(1200), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    with torch.no_grad():
        for img in images:
            print('transforming image {}'.format(img))
            im = Image.open(img).convert('RGB')
            norm_im = transform(im).to(device)
            print('running forward pass')
            o = m.forward(norm_im.unsqueeze(0))
            cls = Image.fromarray((o.detach().squeeze().cpu().numpy()*255).astype('uint8')).resize(im.size, resample=Image.NEAREST)
            cls.save(os.path.splitext(img)[0] + '_nonthresh.png')
            o = denoising_hysteresis_thresh(o.detach().squeeze().cpu().numpy(), thresholds[0], thresholds[1], sigma)
            cls = Image.fromarray((o*255).astype('uint8')).resize(im.size, resample=Image.NEAREST)
            cls.save(os.path.splitext(img)[0] + '_thresh.png')
            print('result extraction')
            # running line vectorization
            lines = vectorize_lines(np.array(cls))
            overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for line in lines:
                draw.line([tuple(x[::-1]) for x in line], width=10, fill=(0, 130, 200, 127))
            del draw
            Image.alpha_composite(im.convert('RGBA'), overlay).save('{}_overlay.png'.format(os.path.splitext(img)[0]))

