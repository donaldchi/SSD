# coding: utf-8

# # ここて行う処理
# - jaccard係数を用いてmacthingを行う
#     - 係数計算は面積のみに注目していて、クラス情報は無視している感じ
# - Hard Negative Miningを行って、data imbalance問題を対応
# - SmoothL1Loss関数を実装

# ## Match 関数
# - 教師データ : loc_t, conf_t_labelを用意していく
# - 教師データ作成のため？ X (そもそも最初から正解ラベルがあるからね)
# - loss関数計算のために、計算結果が正しいものの集計？  O

# ## Hard Negative Mining
# - Negative DBoxに分類されたDBoxのうち、学習に使用するDBoxの数を絞る操作
# - Locationに関する損失値はPositive DBoxにのみ計算できる
#     - Negative DBoxは背景をさしているため、BBoxを持たない
# - Negative DBoxとPositive DBoxデータのimbalance問題を解決する
# - Negative Samplingで優先するDBox
#     - ラベル予測がうまく行っていないDBoxを優先

# ## SmoothL1Loss関数と交差エントロピー誤差関数
#  - matchとHard Negative Mining操作により、損失を計算する際に使用する教師データと予測結果が求まる
#  - 上の結果を用いて損失関数の損失値を計算する
#  - この損失関数は、予測位置と正解位置の間に差が大きい場合、損失値が多くなり、ネットワーク学習が不安定になることを抑えるため、設計されている
#  - ![位置情報の損失関数](../data/loc_loss.png)
#
#  - SmoothL1Loss関数 : 位置情報損失関数 (回帰問題に落とされるため)
#  - 交差エントロピー誤差関数 : 多クラス分類問題

import torch
import torch.nn as nn
import torch.nn.functional as F
from match import match


class SSD_Loss(nn.Module):
    def __init__(self, jaccard_thresh=0.5, negpos_ratio=3, device='cpu'):
        super(SSD_Loss, self).__init__()
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio = negpos_ratio
        self.device = device

    def forward(self, predictions, targets):
        """
        predictions : (location, confidence, self.dbox_list)
        targets: [num_batch, num_object, 5]
        """
        # locations=torch.Size([num_batch, 8732, 4])
        # confidences=torch.Size([num_batch, 8732, 21])
        # dboxes=torch.Size [8732,4])
        locations, confidences, dboxes = predictions

        # locations : [num_batch, 8732, 4]
        # confidences : [num_batch, 8732, 21]
        num_batch = locations.size(0)
        num_dbox = locations.size(1)
        num_classes = confidences.size(2)

        # 正解ラベル : おそらくone hot vectorが格納されている
        # .toは該当deviceに必要なデータタイプにデータを変換してくれる
        pred_labels = torch.LongTensor(num_batch, num_dbox).to(self.device)
        pred_locations = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        for idx in range(num_batch):
            # BBoxの位置情報
            answers = targets[idx][:, :-1].to(self.device)
            # 物体ラベル情報
            labels = targets[idx][:, -1].to(self.device)

            dbox = dboxes.to(self.device)

            # 座標変換を行う時に係数である
            variance = [0.1, 0.2]

            # matcth関数は予測結果が正解かどうかを判定していく
            # 方法:
            #     - BBoxとjaccard係数が一定以上を持つDBoxは当たったことにする
            #      (DBox_location <= BBox_location, DBox_label=BBox_obj_id)
            #     - jaccard係数が閾値以下の場合、バックグラウンドが当たったとして、DBox_labelを
            #       背景idにする。ただ、この場合、位置情報は持たないし、後のloss計算でも使わない
            match(self.jaccard_thresh, answers, dbox, variance, labels, pred_locations, pred_labels, idx)

        # location loss値を計算 : SmoothL1Loss
        # positive結果のlocationのみを使う
        # label = 0 は背景に当たるため
        # torch.Size([num_batch, 8732])
        pos_mask = pred_labels > 0

        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(locations)

        pos_locations = locations[pos_idx].view(-1, 4)
        pos_labels = pred_locations[pos_idx].view(-1, 4)

        loss_location = F.smooth_l1_loss(pos_locations, pos_labels, reduction='sum')

        #  labels loss値を計算 : クロスエントロピー
        batch_confidence = confidences.view(-1, num_classes)

        loss_confidence = F.cross_entropy(batch_confidence, pred_labels.view(-1), reduction='none')

        ##
        # Hard Negative Mining処理
        ##
        num_pos = pos_mask.long().sum(1, keepdim=True)
        loss_confidence = loss_confidence.view(num_batch, -1)  # [num_batch, 8732]
        # 検知に成功したDBoxのloss値は0に設定
        loss_confidence[pos_mask] = 0

        # この部分はまだちょっとわかっていない
        _, loss_idx = loss_confidence.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)
        # 学習をよくするために損失値が大きいNegative DBoxを優先的にとる
        # 上のloss_idx、idx_rankはこの処理のために必要
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # Hard Negative MiningでとってきたNegative DBoxを抽出
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(confidences)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(confidences)

        # [num_pos+num_nge, num_classes]
        # Downsamplingの結果のみを使ってloss値を計算
        confidence_final = confidences[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1, num_classes)
        labels_final = pred_labels[(pos_mask+neg_mask).gt(0)]

        loss_confidence = F.cross_entropy(confidence_final, labels_final, reduction='sum')

        N = num_pos.sum()
        loss_location /= N
        loss_confidence /= N

        return loss_location, loss_confidence
