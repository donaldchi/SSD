# デフォルトボックスの実装
# - default boxの種類
#     - 四つの場合: 大きい正方形、小さい正方形, 縦 : 横 = 1 : 2, 縦 : 横 = 2 : 1
#     - 六つの場合: 縦 : 横 = 3 : 1, 縦 : 横 = 1 : 3 のボックスを上に追加
# - デフォルトボックスサイズの設定が本と違う気がする
#     - 実際には 3:1, 2:1とかになっていない
from itertools import product
from math import sqrt
import torch


class DefaultBox(object):
    def __init__(self, config):
        super(DefaultBox, self).__init__()

        # 初期設定
        self.image_size = config['input_size']

        # location, confidence layerへinputするdataサイズ
        self.input_feature_maps = config['input_feature_maps']
        self.input_len = len(config['input_feature_maps'])
        self.dbox_pixel_sizes = config['dbox_pixel_sizes']
        self.small_dbox_sizes = config['small_dbox_sizes']
        self.big_dbox_sizes = config['big_dbox_sizes']
        self.dbox_aspect_ratios = config['dbox_aspect_ratios']

    def create_dbox_list(self):
        mean = []

        for idx, feature_map_size in enumerate(self.input_feature_maps):
            for feature_map_x, feature_map_y in product(range(feature_map_size), repeat=2):

                # feature_mapを適応後の画像サイズ
                feature_image_size = self.image_size / self.dbox_pixel_sizes[idx]

                # default boxの中心座標
                center_x = (feature_map_x+0.5) / feature_image_size
                center_y = (feature_map_y+0.5) / feature_image_size

                # 小さいdefault box : 中心は変わらない
                small_dbox = self.small_dbox_sizes[idx] / self.image_size
                mean += [center_x, center_y, small_dbox, small_dbox]

                # 大きいdefault box : 中心は変わらない
                big_dbox = sqrt(small_dbox*(self.big_dbox_sizes[idx] / self.image_size))
                mean += [center_x, center_y, big_dbox, big_dbox]

                # 長方形default box
                for dar in self.dbox_aspect_ratios[idx]:
                    mean += [center_x, center_y, small_dbox * sqrt(dar), small_dbox / sqrt(dar)]
                    mean += [center_x, center_y, small_dbox / sqrt(dar), small_dbox * sqrt(dar)]

        print('mean: ', len(mean))
        # default boxを torch.size([8732, 4])に変換
        output = torch.Tensor(mean).view(-1, 4)

        # DBoxが画像の外にはみ出るのを防ぐため、大きさを最小0、最大1にする
        # 1以上の値は全て1に、0以下の値は全てを0に設定する気がする
        output.clamp_(max=1, min=0)

        return output
