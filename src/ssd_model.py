# coding: utf-8
import pandas as pd
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import Image
from l2norm import L2Norm
from dbox import DefaultBox
from non_maximum_suppression import (
    decode,
    nm_suppression
)
from detect import Detect

# # VGG module
# - なぜ34層なのか？　(まだわかっていない)
# - 本が間違っている : 35層ある

# ## 実装の詳細
# - M : 床関数モード floor
# - MC : 天井関数モード ceil
# - conv2d
#     - dilation : a trous algorithmを可能にする
#         - simulation : https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
#     - kernel_size : フィルタサイズ


def vgg():
    layers = []
    in_channels = 3

    layer_conf = [
        64, 64, 'M', 128, 128, 'M',
        256, 256, 256, 'MC',
        512, 512, 512, 'M',
        512, 512, 512
    ]

    for layer in layer_conf:
        if layer == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif layer == 'MC':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            out_channels = layer
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels

    # 5番目のプールは違う設定になる
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # 上ですでに13層のconvを定義済み
    conv14 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv15 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, conv14, nn.ReLU(inplace=True), conv15, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


def extras():
    # # extrasの実装
    # - ReLU関数は順伝播関数の中で用意することにする。何で？

    layers = []
    in_channels = 1024

    #  extrasの設定が間違っている
    #  layer_conf = [512, 512, 256, 256, 256, 256, 512, 512]
    layer_conf = [512, 512, 256, 256, 256, 256, 256, 256]

    layers += [nn.Conv2d(in_channels, layer_conf[0], kernel_size=(1))]
    layers += [nn.Conv2d(layer_conf[0], layer_conf[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(layer_conf[1], layer_conf[2], kernel_size=(1))]
    layers += [nn.Conv2d(layer_conf[2], layer_conf[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(layer_conf[3], layer_conf[4], kernel_size=(1))]
    layers += [nn.Conv2d(layer_conf[4], layer_conf[5], kernel_size=(3))]
    layers += [nn.Conv2d(layer_conf[5], layer_conf[6], kernel_size=(1))]
    layers += [nn.Conv2d(layer_conf[6], layer_conf[7], kernel_size=(3))]

    # 活性化関数のReLUは今回はSSDモデルの順伝搬のなかで用意することにし、
    # extraモジュールでは用意していません

    return nn.ModuleList(layers)


# # LocationとConfidence moduleの実装
# - bbox_aspect_num で各inputで使うbounding boxの数を渡す
# - 本では、この二つのmoduleを一緒に定義してあるが、わかり安くするため分離する
# - input_channels: locationとconfidence layerの入力となるデータのchannel数

def location_layers(input_channels=[512, 1024, 512, 256, 256, 256],
                    bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    location_layers = []

    # locationの座標が4次元のリストになるので、*4になる
    for idx, input_channel in enumerate(input_channels):
        location_layers += [
            nn.Conv2d(input_channel, bbox_aspect_num[idx]*4, kernel_size=3, padding=1)
        ]
    return nn.ModuleList(location_layers)


def confidence_layers(num_classes=21,
                      input_channels=[512, 1024, 512, 256, 256, 256],
                      bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    confidence_layers = []
    for idx, input_channel in enumerate(input_channels):
        confidence_layers += [
            nn.Conv2d(input_channel, bbox_aspect_num[idx]*num_classes, kernel_size=3, padding=1)
        ]
    return nn.ModuleList(confidence_layers)


# SSDクラスの実装

class SSD(nn.Module):
    def __init__(self, phase, config):
        super(SSD, self).__init__()

        self.phase = phase  # train or inference
        self.num_classes = config['num_classes']

        # SSDの組み立て
        self.vgg = vgg()
        self.extras = extras()
        self.L2Norm = L2Norm()

        # num_classes=21,
        # input_channels=[512, 1024, 512, 256, 256, 256],
        # bbox_aspect_num=[4,6,6,6,4,4]
        self.location = location_layers(config['input_channels'],
                                        config['bbox_aspect_num'])
        self.confidence = confidence_layers(config['num_classes'],
                                            config['input_channels'],
                                            config['bbox_aspect_num'])

        # Default Boxの作成
        dbox = DefaultBox(config)
        self.dbox_list = dbox.create_dbox_list()

        if phase == 'inference':
            self.detect = Detect()

    def forward(self, x):
        inputs = list()  # ６種のinputを格納する
        location = list()
        confidence = list()

        for k in range(23):
            x = self.vgg[k](x)

        input1 = self.L2Norm(x)
        inputs.append(input1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        inputs.append(x)

        # input 3 ~ 6 までを抽出
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                inputs.append(x)

        # 各inputに対して、 locationとconfidenceをそれぞれ計算
        # [batch_num, feature_map数, feature_map数, 4*aspect_ratio種類数]
        for (x, l, c) in zip(inputs, self.location, self.confidence):
            location.append(l(x).permute(0,2,3,1).contiguous())
            confidence.append(c(x).permute(0,2,3,1).contiguous())

        # locationのサイズは、torch.Size([batch_num, 34928])
        # confidenceのサイズはtorch.Size([batch_num, 34928*num_classes])になる
        location = torch.cat([o.view(o.size(0), -1) for o in location], 1)
        confidence = torch.cat([o.view(o.size(0), -1) for o in confidence], 1)

        # locationのサイズは、torch.Size([batch_num, 8732, 4])
        # confidenceのサイズは、torch.Size([batch_num, 8732, 21])
        location = location.view(location.size(0), -1, 4)
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        output = (location, confidence, self.dbox_list)

        if self.phase == 'inference':
            return self.detect(output[0], output[1], output[2])
        else:
            return output
