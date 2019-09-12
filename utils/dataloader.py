import cv2
import numpy as np
import os
import random
from threading import Thread
import time
from .augments import *


class ClsDataloader(object):

    def __init__(self, path, img_size=224, batch_size=8, augments=[], balance=0.):
        """分类模型的数据生成器

        Arguments:
            path {str} -- 数据路径, 如'data/mark',其中应包含多个分类文件夹

        Keyword Arguments:
            img_size {int} -- batch中图像的大小 (default: {224})
            batch_size {int} -- 一个batch的大小 (default: {8})
            augments {list} -- augments中的数据增强类的列表 (default: {[]})
            balance {float} -- 样本平衡系数,0为不平衡,1为完全平衡 (default: {0.})
        """
        self.path = path
        self.img_size = img_size
        self.batch_size = batch_size
        self.augments = augments
        self.classes = os.listdir(path)
        self.classes.sort()
        self.data_list = list()

        # 控制样本平衡
        counts = [len(os.listdir(os.path.join(path, c))) for c in self.classes]
        max_count = max(counts)
        for i_c, c in enumerate(self.classes):
            names = os.listdir(os.path.join(path, c))
            if len(names) == 0:
                continue
            new_len = int(((max_count / len(names)) ** balance)*len(names))
            names *= (max_count // len(names))+1
            names = names[:new_len]
            for name in names:
                self.data_list.append([os.path.join(path, c, name), i_c])
        self.iter_times = len(self.data_list) // self.batch_size + 1
        self.max_len = 50
        self.queue = []
        self.scale = img_size
        self.batch_list = []
        t = Thread(target=self.run)
        t.setDaemon(True)
        t.start()

    def __iter__(self):
        return self

    def worker(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.scale, self.scale))
        for aug in self.augments:
            img, _, __ = aug(img)
        return img

    def run(self):
        while True:
            while len(self.batch_list) > self.max_len:
                time.sleep(0.1)
            if len(self.queue) == 0:
                random.shuffle(self.data_list)
                self.queue = self.data_list

            its = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            imgs = [self.worker(it[0]) for it in its]
            self.batch_list.append(
                [np.uint8(imgs), np.int64([it[1] for it in its])])

    def next(self):
        while len(self.batch_list) == 0:
            time.sleep(0.1)
        batch = self.batch_list[0]
        self.batch_list = self.batch_list[1:]
        return batch[0], batch[1]


if __name__ == "__main__":
    augments_list = [
        PerspectiveProject(0, 1),
        HSV_H(0.1, 0.7),
        HSV_S(0.1, 0.7),
        HSV_V(0.1, 0.7),
        # H_Flap(1),
        Rotate(0, 1),
        Blur(0.1, 0.3),
        Noise(0.05, 0.3),
    ]
    d = DetDataloader('data/mark/train.txt',
                      augments=augments_list)
    for i in range(d.iter_times):
        print(d.next())
