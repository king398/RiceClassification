from tqdm.auto import tqdm
import torch
from torch.cuda.amp import autocast
from utils import *
import numpy as np
import gc
from augmentations import *
from loss import *


def train_fn(train_loader, model, criterion, optimizer, epoch, cfg, scheduler=None):
    """Train a model on the given image using the given parameters .
    Args:
        train_loader ([DataLoader]): A pytorch dataloader that containes train images and returns images,target
        model ([Module]): A pytorch model
        criterion ([type]): Pytorch loss
        criterion_2 ([type]): Pytorch loss
        optimizer ([type]): [description]
        epoch ([type]): [description]
        cfg ([type]): [description]
        scheduler ([type], optional): [description]. Defaults to None.
    """
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device(cfg['device'])
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    outputs = None
    targets = None

    for i, (images, target) in enumerate(stream, start=1):

        images = images.to(device, non_blocking=True)
        target = target.to(device).long()

        with autocast():
            output = model(images)
            output = output.float()

        loss = criterion(output, target)

        accuracy = accuracy_score(output, target)
        # log_loss = metric_log_loss(output, target)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        # metric_monitor.update("Log loss", log_loss)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")
        if outputs is None and targets is None:
            outputs = output
            targets = target
        else:
            outputs = torch.cat([outputs, output], dim=0)
            targets = torch.cat([targets, target], dim=0)
    log_loss = metric_log_loss(outputs, targets)
    print(f"Epoch: {epoch:02}. Train. Log Loss {log_loss}")


def validate_fn(val_loader, model, criterion, epoch, cfg):
    device = torch.device(cfg['device'])
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    loss_list = []
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)

            target = target.to(device, non_blocking=True).long()
            with autocast():
                output = model(images).float()
            loss = criterion(output, target)

            log_loss = metric_log_loss(output, target)
            accuracy = accuracy_score(output, target)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            metric_monitor.update("Log loss", log_loss)
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")
            loss_list.append(log_loss)
    return np.mean(loss_list)
