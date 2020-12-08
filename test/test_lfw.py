import argparse
import copy
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tensorflow as tf

from model.mobilefacenet import *


def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def evaluation_10_fold(root='./result/best_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(root)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)
        print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
        print('--------')
        print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
    return ACCs


# parse test list
def parseList(root):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:]
    folder_name = 'lfw-112X96'
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
            fold = i // 600
            flag = 1
        elif len(p) == 4:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
            fold = i // 600
            flag = -1
        nameLs.append(nameL)
        nameRs.append(nameR)
        folds.append(fold)
        flags.append(flag)
    # print(nameLs)
    return [nameLs, nameRs, folds, flags]

# create dataset
def create_dataset(imgl_list, imgr_list):

    def gen():
        for i in range(len(imgl_list)):
            imgl = plt.imread(imgl_list[i])
            if len(imgl.shape) == 2:
                imgl = np.stack([imgl] * 3, 2)
            imgr = plt.imread(imgr_list[i])
            if len(imgr.shape) == 2:
                imgr = np.stack([imgr] * 3, 2)

            # flip image/augmentation
            imglist = [imgl[:, :, :],
                       imgl[:, ::-1, :],
                       imgr[:, :, :],
                       imgr[:, ::-1, :]]
            for j in range(len(imglist)):
                imglist[j] = (imglist[j].astype("float32") - 127.5) / 128.0
            yield tuple(imglist)

    return gen

# get features
def get_features(model, lfw_dir, feature_save_dir, resume=None):

    # load feature generate model
    #model = tf.keras.models.load_model("pretrained_model/inference_model.h5")
    #model = tf.keras.models.load_model("pretrained_model/model_0_ckpt.h5", custom_objects={'ArcFace': ArcFace})
    # inference model save
    #model = tf.keras.models.Model(inputs=model.input[0], outputs=model.layers[-3].output)

    # left ad right features
    featureLs = None
    featureRs = None
    count = 0

    nl, nr, folds, flags = parseList(lfw_dir)
    gen = create_dataset(nl, nr)
    dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32)).batch(32)

    for i, l in enumerate(dataset):

        # feed forward
        res = [model.predict(d) for d in l]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)

        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)

        print(featureRs.shape)
    # save result
    result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
    scipy.io.savemat(feature_save_dir, result)


if __name__ == "__main__":
    lfw_dir = "lfw"
    #nl, nr, folds, flags = parseList(lfw_dir)
    #gen = create_dataset(nl, nr)
    #for l in gen:
    #    print(len(l))
    model = tf.keras.models.load_model("pretrained_model/training_model/replaced_prelu_model.h5")
    get_features(model, lfw_dir, 'result/best_result.mat')
    evaluation_10_fold()
