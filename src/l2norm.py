# # L2Norm層の実装
# - 今回のL2Normは 次元(512, 38, 38)のsource1のtensorに対して行う
# - 38x38個の要素ごとに、512channleにわたる二乗和のルートを計算し、その値を512x38x38の全ての値にそれぞれ割ることで正規化を行う。(説明難しい！)
# - 言うまでもなくこの処理で各特徴量の値の範囲が一定範囲に収まることになる
# - チャンネルごとに学習必要な係数をかけることで重み付けを行う。なぜ？
import torch
import torch.nn as nn
import torch.nn.init as init


class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.init_parameters()
        self.eps = 1e-10

    def init_parameters(self):
        # weightの初期値が全てscaleになる
        init.constant_(self.weight, self.scale)

    def forward(self, x):
        # normのtensorサイズは [bach_num, 1, 38, 38]になる
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        # weightsのサイズが [batch_num, 512, 38, 38]になる
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        # print(weights.shape)
        out = weights * x

        return out
