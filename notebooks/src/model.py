import timm
import torch.nn as nn
import torch


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class BaseModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg['model'], pretrained=self.cfg['pretrained'],
                                       in_chans=self.cfg['in_channels'],
                                       num_classes=cfg['target_size'])
        self.model = self.model.apply(set_batchnorm_eval)

    def forward(self, x):
        output = self.model(x)

        return output


class BaseModelFeature(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg['model'], pretrained=self.cfg['pretrained'],
                                       in_chans=self.cfg['in_channels'],
                                       num_classes=0)
        self.model = self.model.apply(set_batchnorm_eval)
        self.fc = nn.LazyLinear(self.cfg['target_size'])

    def forward(self, x):
        feature = self.model(x)
        output = self.fc(feature)
        return output, feature
