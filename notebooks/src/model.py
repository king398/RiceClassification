import timm
import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg['model'], pretrained=self.cfg['pretrained'],
                                       in_chans=self.cfg['in_channels'],
                                       num_classes=0)
        self.fc = nn.LazyLinear(3)

    def set_batchnorm_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def forward(self, x, x_rgn):
        output = self.model(x)
        output_1 = self.model(x_rgn)
        output = self.fc(torch.cat([output_1, output], 1))

        return output
