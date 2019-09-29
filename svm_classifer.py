import cv2
import numpy as np
import os
from cv_utils.processors import *
from cv_utils.utils import *
from tqdm import tqdm



def train_svm(train_dir, val_dir, C=1e-3, gamma=1, kernel=cv2.ml.SVM_LINEAR):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(kernel)
    svm.setC(C)
    svm.setGamma(gamma)
    processor_list = [
        ComputeHog(),
        SobelX(),
        SobelY(),
        Resize(size=(16, 16))
    ]
    inputs, targets = simple_dataloader(train_dir,
                                processor_list=processor_list
                                )
    svm.train(inputs, cv2.ml.ROW_SAMPLE, targets)
    inputs, targets = simple_dataloader(val_dir,
                                processor_list=processor_list
                                )
    _, pred = svm.predict(inputs)
    pred = pred[:, 0]
    for i in range(15):
        targets_c = targets[pred == i]
        pred_c = pred[pred == i]
        tp = np.sum(pred_c == targets_c)
        print(i, tp/len(pred_c))
    tp = np.sum(pred == targets)
    print(tp/len(pred))
    return svm


if __name__ == "__main__":
    train_dir = 'data/road_mark/train'
    val_dir = 'data/road_mark/val'
    # for c in range(-5,5):
    #     for g in range(-5,5):
    #         svm = train_svm(train_dir, val_dir, 10**c, 10**g, cv2.ml.SVM_RBF)
    svm = train_svm(train_dir, val_dir, 10, 0.01, cv2.ml.SVM_LINEAR)
    svm.save('svm.yaml')
