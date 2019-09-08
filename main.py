import cv2
import numpy as np
import os

train_path = 'data/train'
test_path = 'data/test'


def compute_hog(img):
    winSize = (8, 8)
    blockSize = (8, 8)
    blockStride = (8, 8)
    cellSize = (4, 4)
    nbins = 5
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


def load_data(path, img_size=None, compute_func_list=[]):
    inputs = []
    targets = []
    classes = os.listdir(path)
    for ci, c in enumerate(classes):
        names = os.listdir(os.path.join(path, c))
        for name in names:
            features = []
            img = cv2.imread(os.path.join(path, c, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_size is not None:
                img = cv2.resize(img, img_szie)
            for func in compute_func_list:
                feature = func(img.copy())
                feature = normalize(feature).reshape(-1)
                features.append(feature)
            img = np.float32(img)
            img = normalize(img)
            img = img.reshape(-1)
            features.append(img)
            features = np.concatenate(features)
            inputs.append(features)
            targets.append(ci)
    inputs = np.float32(inputs)
    targets = np.int64(targets)
    return inputs, targets


def train_svm(C=1e-3, gamma=1, kernel=cv2.ml.SVM_LINEAR):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(kernel)
    svm.setC(C)
    svm.setGamma(gamma)

    inputs, targets = load_data(train_path,
                                compute_func_list=[compute_hog]
                                )
    svm.train(inputs, cv2.ml.ROW_SAMPLE, targets)
    inputs, targets = load_data(test_path,
                                compute_func_list=[compute_hog]
                                )
    _, pred = svm.predict(inputs)
    pred = pred[:, 0]
    tp = np.sum(pred == targets)
    print(tp/len(pred))
    return svm


if __name__ == "__main__":
    svm = train_svm(10, 0.01, cv2.ml.SVM_RBF)
    svm.save('gray_hog.xml')
