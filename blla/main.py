#!/usr/bin/env python3

import os
import glob
import json
import click
import torch

from os import path
from torch import nn, optim
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import DataLoader

from blla.model import VerticeNet, PolyLineNet, FeatureNet
from blla.darknet import Darknet53
from blla.dataset import InitialVertexDataset, VerticesDataset
from blla.postprocess import denoising_hysteresis_thresh

@click.group()
def cli():
    pass

@cli.command()
@click.option('-i', '--load', default=None, type=click.Path(exists=True, readable=True), help='pretrained weights to load')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.argument('ground_truth', nargs=1)
def compute_features(load, device, ground_truth):
    click.echo('instantiating darknet network')
    dk = Darknet53().to(device)
    click.echo('loading darknet weights')
    dk.load_state_dict(torch.load(load, map_location=device))
    click.echo('instantiating feature network')
    net = FeatureNet(dk)

    click.echo('writing features')
    tfs = transforms.Compose([transforms.Resize(900), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    with torch.no_grad():
        for p in glob.glob('{}/**/*.plain.png'.format(ground_truth), recursive=True):
            with Image.open(p) as img:
                click.echo(p)
                with open(path.splitext(path.splitext(p)[0])[0] + '.feat', 'wb') as fp:
                    i = tfs(imgs).unsqueeze(0).to(device)
                    o = net(i).squeeze()
                    torch.save(o, fp)


@cli.command()
@click.option('-n', '--name', default=None, help='prefix for checkpoint file names')
@click.option('-i', '--load', default=None, type=click.Path(exists=True, readable=True), help='pretrained weights to load')
@click.option('-l', '--lrate', default=2e-4, help='initial learning rate')
@click.option('--weight-decay', default=1e-5, help='weight decay')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-v', '--validation', default='val', help='validation set location')
@click.argument('ground_truth', nargs=1)
def ivtrain(name, load, lrate, weight_decay, workers, device, validation, ground_truth):

    if not name:
        name = '{}_{}'.format(lrate, weight_decay)
    click.echo('model output name: {}'.format(name))

    torch.set_num_threads(1)

    train_set = InitialVertexDataset(glob.glob('{}/**/*.lines.json'.format(ground_truth), recursive=True))
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=1, shuffle=True, pin_memory=True)
    val_set = InitialVertexDataset(glob.glob('{}/**/*.lines.json'.format(validation), recursive=True))
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=1, pin_memory=True)

    click.echo('loading network')
    model = ResUNet(refine_encoder=False).to(device)

    if load:
        click.echo('loading weights')
        model = torch.load(load, map_location=device)

    criterion = nn.BCEWithLogitsLoss()
    opti = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)

    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    def output_preprocess(output):
        o, target = output
        o = torch.sigmoid(o)
        o = denoising_hysteresis_thresh(o.detach().squeeze().cpu().numpy(), 0.8, 0.9, 2.5)
        return torch.from_numpy(o.astype('f')).unsqueeze(0).unsqueeze(0).to(device), target.double().to(device)

    trainer = create_supervised_trainer(model, opti, criterion, device=device, non_blocking=True)
    evaluator = create_supervised_evaluator(model, device=device, non_blocking=True, metrics={'accuracy': Accuracy(output_transform=output_preprocess),
                                                                                              'precision': Precision(output_transform=output_preprocess),
                                                                                              'recall': Recall(output_transform=output_preprocess),
                                                                                              'loss': Loss(criterion)})
    ckpt_handler = ModelCheckpoint('.', name, save_interval=1, n_saved=10, require_empty=False)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, ['loss'])

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=ckpt_handler, to_save={'net': model})
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=TerminateOnNan())

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_data_loader)
        metrics = evaluator.state.metrics
        progress_bar.log_message('eval results - epoch {} loss: {:.4f} accuracy: {:.4f} recall: {:.4f} precision {:.4f}'.format(engine.state.epoch,
                                                                                                                   metrics['loss'],
                                                                                                                   metrics['accuracy'],
                                                                                                                   metrics['recall'],
                                                                                                                   metrics['precision']))
    trainer.run(train_data_loader, max_epochs=1000)

