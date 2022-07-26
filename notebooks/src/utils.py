import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, CosineAnnealingLR
from collections import defaultdict
from sklearn.metrics import log_loss


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def return_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def accuracy_score(output, labels):
    output = output.detach().cpu()
    labels = labels.detach().cpu()
    output = torch.softmax(output, 1)
    accuracy = (output.argmax(dim=1) == labels).float().mean()
    accuracy = accuracy.detach().cpu().numpy()
    return accuracy


def metric_log_loss(output, labels):
    output = output.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    accuracy = log_loss(labels, output)
    return accuracy


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def get_scheduler(optimizer, scheduler_params, train_loader=None):
    if scheduler_params['scheduler_name'] == 'CosineAnnealingWarmRestarts':
        schedulers = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params['T_0'],
            eta_min=scheduler_params['min_lr'],
            last_epoch=-1
        )

    elif scheduler_params['scheduler_name'] == 'OneCycleLR':
        schedulers = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                         epochs=scheduler_params['epochs'],
                                                         steps_per_epoch=len(train_loader),
                                                         max_lr=scheduler_params['max_lr'],
                                                         pct_start=scheduler_params['pct_start'],

                                                         )
    elif scheduler_params['scheduler_name'] == 'CosineAnnealingLR':
        schedulers = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params['T_max'],
            eta_min=scheduler_params['min_lr'],
            last_epoch=-1
        )
    return schedulers


def return_filpath(name, folder):
    path = os.path.join(folder, f'{name}')
    return path


def return_label(blast, brown, healthy):
    probablity = np.array([blast, brown, healthy])
    return np.argmax(probablity)
