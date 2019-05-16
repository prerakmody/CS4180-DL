#encoding:utf-8
import sys
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.nn.functional as F


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def getYOLOv1(name=''):
    if name != '':
        myYOLO            = YOLOv1(name, cfg['D'], batch_norm=True)
        VGG               = models.vgg16_bn(pretrained=True)
        state_dict_VGG    = VGG.state_dict()
        state_dict_myYOLO = myYOLO.state_dict()
        
        for k in state_dict_VGG.keys():
            if k in state_dict_myYOLO.keys() and k.startswith('features'):
                state_dict_myYOLO[k] = state_dict_VGG[k]
        myYOLO.load_state_dict(state_dict_myYOLO)
        return myYOLO

    else:
        print (' - Pass a name for your model')
        sys.exit(1)
    


class YOLOv1(nn.Module):

    def __init__(self, name, cfg, batch_norm, image_size=448):
        super(YOLOv1, self).__init__()
        self.name       = name
        self.features   = self.getFeatureLayers(cfg, batch_norm)
        self.classifier = nn.Sequential( # add the regression part to the features
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1470),
        )
        self._initialize_weights()
        self.image_size = image_size

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        x = x.view(-1,7,7,30)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def getFeatureLayers(self, cfg, batch_norm=False):
        if (1):
            params_in_channels  = 3
            params_conv_stride  = 1
            params_conv_size    = 3 
            params_first_flag   = True
            params_pool_stride  = 2
            params_pool_kernel  = 2

        layers = []
        for item in cfg:
            params_conv_stride = 1
            if (item == 64 and params_first_flag):
                params_conv_stride = 2
                params_first_flag  = False

            if item == 'M': # max-pooling
                layers += [nn.MaxPool2d(kernel_size=params_pool_kernel, stride=params_pool_stride)]
            else:
                params_kernels = item
                conv2d = nn.Conv2d(params_in_channels, params_kernels, kernel_size=params_conv_size, stride=params_conv_stride, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(item), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                params_in_channels = item
        return nn.Sequential(*layers)


def test():
    net = getYOLOv1()
    img = torch.rand(1,3,448,448)
    img = Variable(img)
    output = net(img)
    print(output.size())

# if __name__ == '__main__':
#     test()