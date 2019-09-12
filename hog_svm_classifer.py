import cv2
import numpy as np
import os
from utils.dataloader import ClsDataloader
from utils.augments import *
from tqdm import tqdm


def compute_hog(img):
    winSize = (16, 16)
    blockSize = (8, 8)
    blockStride = (2, 2)
    cellSize = (4, 4)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                            winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    descriptor = hog.compute(img)
    return descriptor[:, 0]


def normalize(data):
    data -= np.min(data)
    if np.max(data) > 0:
        data /= np.max(data)
    return data


def get_features(img, img_size=None, compute_func_list=[]):
    features = []
    if img_size is not None:
        img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for func in compute_func_list:
        feature = func(img.copy())
        feature = normalize(feature).reshape(-1)
        features.append(feature)
    img = np.float32(img)
    img = normalize(img)
    img = img.reshape(-1)
    features.append(img)
    features = np.concatenate(features)
    return features


def load_data(path, img_size=None, augments_list=[], compute_func_list=[]):
    inputs = []
    targets = []
    dataloader = ClsDataloader(
        path, batch_size=1, augments=augments_list, balance=0.5)
    for it in tqdm(range(dataloader.iter_times)):
        i, t = dataloader.next()
        inputs.append(get_features(i[0], img_size, compute_func_list))
        targets.append(t[0])
    inputs = np.float32(inputs)
    targets = np.int64(targets)
    return inputs, targets


def train_svm(train_dir, val_dir, C=1e-3, gamma=1, kernel=cv2.ml.SVM_LINEAR):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(kernel)
    svm.setC(C)
    svm.setGamma(gamma)
    augments_list = [
        PerspectiveProject(0.1, 0.3),
        HSV_H(0.1, 0.3),
        HSV_S(0.1, 0.3),
        HSV_V(0.1, 0.3),
        # Rotate(1, 0.3),
        Blur(0.1, 0.3),
        Noise(0.05, 0.3),
    ]
    # augments_list = []
    inputs, targets = load_data(train_dir, (32, 32), augments_list, 
                                compute_func_list=[compute_hog]
                                )
    svm.train(inputs, cv2.ml.ROW_SAMPLE, targets)
    inputs, targets = load_data(val_dir, (32, 32),
                                compute_func_list=[compute_hog]
                                )
    _, pred = svm.predict(inputs)
    pred = pred[:, 0]
    for i in range(15):
        targets_c = targets[pred==i]
        pred_c = pred[pred==i]
        tp = np.sum(pred_c == targets_c)
        print(i, tp/len(pred_c))
    tp = np.sum(pred == targets)
    print(tp/len(pred))
    return svm


if __name__ == "__main__":
    train_dir = 'data/road_mark_with_orientation/train'
    val_dir = 'data/road_mark_with_orientation/val'
    # for c in range(-5,5):
    #     for g in range(-5,5):
    #         svm = train_svm(train_dir, val_dir, 10**c, 10**g, cv2.ml.SVM_RBF)
    svm = train_svm(train_dir, val_dir, 10, 0.01, cv2.ml.SVM_LINEAR)
    svm.save('svm.xml')
