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

from blla.model import VerticeNet, PolyLineNet, FeatureNet, FeatureExtractionNet
from blla.darknet import Darknet53
from blla.dataset import InitialVertexDataset, VerticesDataset
from blla.postprocess import hysteresis_thresh

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
    net = FeatureExtractionNet(dk)

    click.echo('writing features')
    tfs = transforms.Compose([transforms.Resize(900), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    with torch.no_grad():
        for p in glob.glob('{}/**/*.plain.png'.format(ground_truth), recursive=True):
            with Image.open(p) as img:
                click.echo(p)
                with open(path.splitext(path.splitext(p)[0])[0] + '.feat', 'wb') as fp:
                    i = tfs(img).unsqueeze(0).to(device)
                    ds_2, ds_3, ds_4, ds_5, ds_6 = net(i)
                    ds_2 = ds_2.to('cpu')
                    ds_3 = ds_3.to('cpu')
                    ds_4 = ds_4.to('cpu')
                    ds_5 = ds_5.to('cpu')
                    ds_6 = ds_6.to('cpu')
                    torch.save({'ds_2': ds_2.squeeze(), 'ds_3': ds_3.squeeze(), 'ds_4': ds_4.squeeze(), 'ds_5': ds_5.squeeze(), 'ds_6': ds_6.squeeze()}, fp)


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

    train_set = InitialVertexDataset(glob.glob('{}/**/*.plain.png'.format(ground_truth), recursive=True))
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=1, shuffle=True, pin_memory=True)
    val_set = InitialVertexDataset(glob.glob('{}/**/*.plain.png'.format(validation), recursive=True))
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=1, pin_memory=True)

    click.echo('loading network')
    model = VerticeNet().to(device)

    if load:
        click.echo('loading weights')
        model = torch.load(load, map_location=device)

    criterion = nn.BCEWithLogitsLoss()
    opti = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)

    for epoch in range(999):
        epoch_loss = 0
        with click.progressbar(train_data_loader, label='epoch {}'.format(epoch), show_pos=True) as bar:
            for batch in bar:
                ds_2, ds_3, ds_4, ds_5, ds_6, target = batch
                ds_2 = ds_2.to(device, non_blocking=True)
                ds_3 = ds_3.to(device, non_blocking=True)
                ds_4 = ds_4.to(device, non_blocking=True)
                ds_5 = ds_5.to(device, non_blocking=True)
                ds_6 = ds_6.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                opti.zero_grad()
                o = model(ds_2, ds_3, ds_4, ds_5, ds_6).squeeze(0)
                loss = criterion(o, target)
                epoch_loss += loss
                loss.backward()
                opti.step()
        torch.save(model.state_dict(), '{}_{}.pth'.format(name, epoch))
        print("===> epoch {} complete: avg. loss: {:.4f}".format(epoch, epoch_loss / len(train_data_loader)))
        val_acc, val_recall, val_precision, val_loss = evaluate(model, device, criterion, val_data_loader)
        model.train()
        print("===> epoch {} validation loss: {:.4f} (accuracy: {:.4f}, recall: {:.4f}, precision: {:.4f})".format(epoch, val_loss, val_acc, val_recall, val_precision))


def evaluate(model, device, criterion, data_loader):
    model.eval()
    accuracy = 0.0
    recall = 0.0
    precision = 0.0
    loss = 0.0
    with torch.no_grad():
        for sample in data_loader:
            ds_2, ds_3, ds_4, ds_5, ds_6, target = sample
            ds_2 = ds_2.to(device, non_blocking=True)
            ds_3 = ds_3.to(device, non_blocking=True)
            ds_4 = ds_4.to(device, non_blocking=True)
            ds_5 = ds_5.to(device, non_blocking=True)
            ds_6 = ds_6.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            o = model(ds_2, ds_3, ds_4, ds_5, ds_6).squeeze(0)
            loss += criterion(o, target)
            o = torch.sigmoid(o)
            pred = hysteresis_thresh(o.detach().squeeze().cpu().numpy(), 0.5, 0.5)
            tp = float((pred == target.detach().squeeze().cpu().numpy()).sum())
            recall += tp / target.sum()
            precision += tp / pred.sum()
            accuracy += tp / len(target.view(-1))
    return accuracy / len(data_loader), recall / len(data_loader), precision / len(data_loader), loss / len(data_loader)

