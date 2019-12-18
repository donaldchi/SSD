# # Detect クラスの実装
# - Bounding Boxの情報を得る [x_min, y_min, x_max, y_max]
# - confidenceが一定以上のBounding Boxを取り出す
# - Non-Maximum Suppressionを行い、最終的な検知Bounding Boxを残す
from torch.autograd import Function
import torch.nn as nn


class Detect(Function):
    def __init__(self, confidence_thresh=0.01, top_k=200, nms_thresh=0.45):
        # confidenceの値を正規化する時に使う
        self.softmax = nn.Softmax(dim=-1)
        self.confidence_thresh = confidence_thresh
        self.top_k = top_k

        # 　Bounding Boxで被る面積がこの閾値以上の場合、Non-Maximum Suppression
        #  の処理対象と考える
        self.nms_thresh = nms_thresh

    def forward(self, location_data, confidence_data, dbox_list):
        """
        順伝播の計算を実行する

        :return:
        output : torch.Size([batch_num, 21, 200, 5])
        (batch_num, クラス, top200のconfidence, Bouding Boxの情報)
        """

        num_batch = location_data.size(0)
        # Default Boxの数 : 8732
        num_dbox = location_data.size(1)
        # 予測クラスの数 : 21
        num_classes = confidence_data.size(2)

        confidence_data = self.softmax(confidence_data)

        # なぜ5？ [x_min, y_min, x_max, y_max, object_index]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # confidence_dataを[batch_num,8732,num_classes]から[batch_num, num_classes,8732]に順番変更
        confidence_detections = confidence_data.transpose(2, 1)

        for i in range(num_batch):
            # Bounding Boxの位置情報の計算
            decoded_boxes = decode(location_data[i], dbox_list)
            confidence_scores = confidence_detections[i].clone()

            for cl in range(1, num_classes):
                # 0は背景クラスになるため無視

                # 閾値を超えたものを1に、以下のものを0にする処理
                #  この処理の結果を使って、閾値を超えたBouding Boxのフィルタリングを行う
                # confidence_scores:torch.Size([21, 8732])
                # c_mask:torch.Size([8732])
                c_mask = confidence_scores[cl].gt(self.confidence_thresh)

                # scoresはtorch.Size([閾値を超えたBBox数])
                scores = conf_scores[cl][c_mask]

                # 閾値を超えたconfがない場合、つまりscores=[]のときは、何もしない
                if scores.nelement() == 0:  # nelementで要素数の合計を求める
                    continue

                # c_maskを、decoded_boxesに適用できるようにサイズを変更します
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask:torch.Size([8732, 4])

                # l_maskをdecoded_boxesに適応します
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # decoded_boxes[l_mask]で1次元になってしまうので、
                # viewで（閾値を超えたBBox数, 4）サイズに変形しなおす

                # Non-Maximum Suppressionを実施
                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)
                # ids：confの降順にNon-Maximum Suppressionを通過したindexが格納
                # count：Non-Maximum Suppressionを通過したBBoxの数

                # outputにNon-Maximum Suppressionを抜けた結果を格納
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)

        return output  # torch.Size([1, 21, 200, 5])
