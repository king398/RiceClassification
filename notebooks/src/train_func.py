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
    if epoch < 5:
        cfg['mixup'] = True
    else:
        cfg['mixup'] = False

    for i, (images, target) in enumerate(stream, start=1):

        images = images.to(device, non_blocking=True)
        target = target.to(device).long()
        if cfg['mixup']:
            images, target_a, target_b, lam = cutmix(images, target, cfg['mixup_alpha'])
            target_a = target_a.to(device)
            target_b = target_b.to(device)

        with autocast():
            output = model(images)
            output = output.float()
        if cfg['mixup']:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            loss = criterion(output, target)

        accuracy = accuracy_score(output, target)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
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


def validate_fn(val_loader, model, criterion, epoch, cfg):
    device = torch.device(cfg['device'])
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    outputs = None
    targets = None
    with torch.no_grad():
        for i, (image, target) in enumerate(stream, start=1):

            images = images.to(device, non_blocking=True)
            target = target.to(device).long()

            with autocast():
                output = model(images)
                output = output.float()

            loss = criterion(output, target)

            accuracy = accuracy_score(output, target)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")
            if outputs is None and targets is None:
                outputs = output
                targets = target
            else:
                outputs = torch.cat([outputs, output], dim=0)
                targets = torch.cat([targets, target], dim=0)
    log_loss = metric_log_loss(outputs, targets)
    print(F"Epoch: {epoch:02}. Valid. Log Loss {str(round(log_loss, 6))}")
    return log_loss


def inference_fn(test_loader, model, cfg):
    device = torch.device(cfg['device'])
    model.eval()
    stream = tqdm(test_loader)
    preds = None
    with torch.no_grad():
        for i, (images) in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)

            with autocast():
                output = model(images)
                output = output.float()

            pred = torch.log_softmax(output, 1).detach().cpu()
            if preds is None:
                preds = pred
            else:
                preds = torch.cat((preds, pred))
            del pred
            gc.collect()

    return preds


def oof_fn(test_loader, model, cfg):
    device = torch.device(cfg['device'])
    model.eval()
    stream = tqdm(test_loader)
    ids = []
    target = []
    preds = None
    probablitys = None
    accuracy_list = []
    with torch.no_grad():
        for i, (images, label, id) in enumerate(stream, start=1):
            ids.extend(id)
            target.extend(label)
            images = images.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).long()
            with autocast():
                output = model(images).float()
            probablity = torch.softmax(output, 1).detach().cpu()
            if probablitys is None:
                probablitys = probablity
            else:
                torch.cat((probablitys, probablity))
            pred = torch.argmax(output, 1).detach().cpu()
            if preds is None:
                preds = pred
            else:
                preds = torch.cat((preds, pred))
            accuracy_list.append(accuracy_score(output, label))
    return ids, target, preds.detach().cpu().numpy(), probablitys.detach().cpu().numpy(), np.mean(accuracy_list)
