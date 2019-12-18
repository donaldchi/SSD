# coding: utf-8
import os.path as os_path
import random
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from xml2list import Anno_xml2list
from data_transformer import DataTransform
from voc_dataset import VOCDataset

torch.manual_seed(20191208)
np.random.seed(20191208)
random.seed(20191208)


# データ読み込みの準備
def _get_img_and_anno_names_list(file_names, path):
    img_path_template = os_path.join(path, 'JPEGImages', '{}.jpg')
    anno_path_template = os_path.join(path, 'Annotations', '{}.xml')

    img_list = []
    anno_list = []

    for file_name in open(file_names):
        img_path = img_path_template.format(file_name.strip())
        anno_path = anno_path_template.format(file_name.strip())

        img_list.append(img_path)
        anno_list.append(anno_path)

    return img_list, anno_list


def _get_data_file_names(path):
    train_file_names = '{}/ImageSets/Main/train.txt'.format(path)
    val_file_names = '{}/ImageSets/Main/val.txt'.format(path)

    train_img_names, train_anno_names = _get_img_and_anno_names_list(train_file_names, path)
    val_img_names, val_anno_names = _get_img_and_anno_names_list(val_file_names, path)

    return train_img_names, train_anno_names, val_img_names, val_anno_names

# # DataLoaderの作成
# - 画像ごとにgtの数が違う, gtは5次元のlistになっているため、自前のDataLoadを作成する必要がある
# - データセットから取り出す変数のサイズがデータごとに異なる場合、自前のDataLoaderが必要になる


def _od_collate_fn(batch):
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なります。
    画像内の物体数が2個であれば(2, 5)というサイズですが、3個であれば（3, 5）など変化します。
    この変化に対応したDataLoaderを作成するために、
    カスタイマイズした、collate_fnを作成します。
    collate_fnは、PyTorchでリストからmini-batchを作成する関数です。
    ミニバッチ分の画像が並んでいるリスト変数batchに、
    ミニバッチ番号を指定する次元を先頭に1つ追加して、リストの形を変形します。
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0] は画像imgです
        targets.append(torch.FloatTensor(sample[1]))  # sample[1] はアノテーションgtです

    # imgsはミニバッチサイズのリストになっています
    # リストの要素はtorch.Size([3, 300, 300])です。
    # このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換します
    imgs = torch.stack(imgs, dim=0)

    # targetsはアノテーションデータの正解であるgtのリストです。
    # リストのサイズはミニバッチサイズです。
    # リストtargetsの要素は [n, 5] となっています。
    # nは画像ごとに異なり、画像内にある物体の数となります。
    # 5は [xmin, ymin, xmax, ymax, class_index] です

    return imgs, targets


def create_dataloader(path, voc_classes, batch_size = 32):
    train_img_names, train_anno_names, val_img_names, val_anno_names = _get_data_file_names(path)

    color_mean = (104, 117, 123)  # (BGR)の色の平均値
    input_size = 300  # 画像のinputサイズを300×300にする

    train_dataset = VOCDataset(train_img_names, train_anno_names, phase="train", transform=DataTransform(
        input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

    val_dataset = VOCDataset(val_img_names, val_anno_names, phase="val", transform=DataTransform(
        input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

    # DataLoaderを作成する
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_od_collate_fn)

    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_od_collate_fn)

    # 辞書オブジェクトにまとめる
    return {"train": train_dataloader, "val": val_dataloader}
