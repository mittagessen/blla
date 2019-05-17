#!/usr/bin/env python3

import os
import glob
import json
import click
import torch

from torch import nn, optim
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import DataLoader

from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, TerminateOnNan
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage, Loss
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator

from blla.model import ResUNet, RecLabelNet
from blla.dataset import BaselineSet
from blla.postprocess import denoising_hysteresis_thresh

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
@click.argument('ground_truth', nargs=1)
def train(name, load, lrate, weight_decay, workers, smooth, device, validation, ground_truth):

    if not name:
        name = '{}_{}'.format(lrate, weight_decay)
    click.echo('model output name: {}'.format(name))

    torch.set_num_threads(1)

    train_set = BaselineSet(glob.glob('{}/**/*.seeds.png'.format(ground_truth), recursive=True), smooth=smooth)
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=1, shuffle=True, pin_memory=True)
    val_set = BaselineSet(glob.glob('{}/**/*.seeds.png'.format(validation), recursive=True), smooth=False)
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=1, pin_memory=True)

    click.echo('loading network')
    model = RecLabelNet().to(device)

    if load:
        click.echo('loading weights')
        model = torch.load(load, map_location=device)

    criterion = nn.MSELoss()
    opti = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)

    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    def output_preprocess(output):
        o, target = output
        o = denoising_hysteresis_thresh(o.detach().squeeze().cpu().numpy(), 0.4, 0.5, 0)
        return torch.from_numpy(o.astype('f')).unsqueeze(0).unsqueeze(0).to(device), target.double().to(device)

    trainer = create_supervised_trainer(model, opti, criterion, device=device, non_blocking=True)
    accuracy = Accuracy(output_transform=output_preprocess)
    precision = Precision(output_transform=output_preprocess)
    recall = Recall(output_transform=output_preprocess)
    loss = Loss(criterion)
    evaluator = create_supervised_evaluator(model, device=device, non_blocking=True)

    accuracy.attach(evaluator, 'accuracy')
    precision.attach(evaluator, 'precision')
    recall.attach(evaluator, 'recall')
    loss.attach(evaluator, 'loss')

    ckpt_handler = ModelCheckpoint('.', name, save_interval=1, n_saved=100, require_empty=False)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, ['loss'])

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=ckpt_handler, to_save={'net': model})
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
    trainer.run(train_data_loader, max_epochs=1000)

