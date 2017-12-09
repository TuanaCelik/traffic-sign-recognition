#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, absolute_import
from __future__ import division

import six.moves.urllib.request as request
import six
import os
import tarfile
import shutil
import hashlib
import sys

import cPickle

import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        return cPickle.load(fo)


def oneHotVector(classIdx, numClasses):
    v = np.zeros((len(classIdx), numClasses), dtype=np.int)
    v[np.arange(0, len(v)), classIdx] = 1
    return v


class gtsrb:
    IMG_WIDTH = 32
    IMG_HEIGHT = 32
    IMG_CHANNELS = 3
    CLASS_COUNT = 43

    dataPath = ''
    batchSize = 100
    trainData = np.array([])
    trainLabels = np.array([])
    testData = np.array([])
    testLabels = np.array([])
    currentIndexTrain = 0
    currentIndexTest = 0
    nTrainSamples = 0
    nTestSamples = 0
    pTrain = []
    pTest = []

    def __init__(self, batchSize=100):
        self.data = np.load('gtsrb_dataset.npz')
        self.batchSize = batchSize
        #self.dataPath = _maybe_download_cifar10(download_dir=downloadDir, download_url=downloadUrl)
        self.loadGTSRB()

        self.trainData = self._swapChannelOrdering(self.trainData)
        self.testData = self._swapChannelOrdering(self.testData)

    def preprocess(self, channels=3):
        print('preprocessing')
        #print(images.shape())
        #print(self.trainData.shape)
        trainData = np.reshape(self.trainData, [-1, 32, 32, 3])
        testData  = np.reshape(self.testData, [-1, 32, 32, 3])
        print(self.trainData.shape)
        for i in range(0,2) :
            mean_channel_train = np.mean(trainData[:, :, :, i])
            mean_channel_test = np.mean(testData[:, :, :, i])
            stddev_channel_train = np.std(trainData[:, :, :, i])
            stddev_channel_test = np.std(testData[:, :, :, i])
            trainData[:, :, :, i] = (trainData[:, :,:, i]  - mean_channel_train) / stddev_channel_train
            testData[:, :, :, i] = (testData[:, :,:, i]  - mean_channel_test) / stddev_channel_test
        
        self.trainData = np.reshape(trainData, [-1, 32*32*3])
        self.testData = np.reshape(testData, [-1, 32*32*3])
        #return images

    def _normaliseImages(self, imgs_flat):
        min = np.min(imgs_flat)
        max = np.max(imgs_flat)
        range = max - min
        return (imgs_flat - min) / range

    def _unflatten(self, imgs_flat):
        return imgs_flat.reshape(-1, self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS)

    def _flatten(self, imgs):
        return imgs.reshape(-1, self.IMG_WIDTH * self.IMG_HEIGHT * self.IMG_CHANNELS)

    def _swapChannelOrdering(self, imgs_flat):
        return self._flatten(imgs_flat.reshape(-1, self.IMG_CHANNELS, self.IMG_WIDTH, self.IMG_HEIGHT)\
                             .transpose(0, 2, 3, 1))

    def loadGTSRB(self):
        
        self.trainData = self.data['X_train']
        self.trainLabels = self.data['y_train']
        self.testData = self.data['X_test']
        self.testLabels = self.data['y_test']

        self.nTrainSamples = len(self.trainLabels)
        self.nTestSamples = len(self.testLabels)

        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)

    def getTrainBatch(self, allowSmallerBatches=False):
        return self._getBatch('train', allowSmallerBatches)

    def getTestBatch(self, allowSmallerBatches=False):
        return self._getBatch('test', allowSmallerBatches)

    def _getBatch(self, dataSet, allowSmallerBatches=False):
        D = np.array([])
        L = np.array([])

        if dataSet == 'train':
            train = True
            test = False
        elif dataSet == 'test':
            train = False
            test = True
        else:
            raise ValueError('_getBatch: Unrecognised set: ' + dataSet)

        while True:
            if train:
                r = range(self.currentIndexTrain,
                          min(self.currentIndexTrain + self.batchSize - L.shape[0], self.nTrainSamples))
                self.currentIndexTrain = r[-1] + 1 if r[-1] < self.nTrainSamples - 1 else 0
                (d, l) = (self.trainData[self.pTrain[r]][:], self.trainLabels[self.pTrain[r]][:])
            elif test:
                r = range(self.currentIndexTest,
                          min(self.currentIndexTest + self.batchSize - L.shape[0], self.nTestSamples))
                self.currentIndexTest = r[-1] + 1 if r[-1] < self.nTestSamples - 1 else 0
                (d, l) = (self.testData[self.pTest[r]][:], self.testLabels[self.pTest[r]][:])

            if D.size == 0:
                D = d
                L = l
            else:
                D = np.concatenate((D, d))
                L = np.concatenate((L, l))

            if D.shape[0] == self.batchSize or allowSmallerBatches:
                break

        return (D, L)

    def showImage(self, image):
        from matplotlib import pyplot as plt
        plt.imshow(image.reshape((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)), interpolation='nearest')
        plt.show()

    def reset(self):
        self.currentIndexTrain = 0
        self.currentIndexTest = 0
        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)


if __name__ == '__main__':
    gtsrb = gtsrb(batchSize=128)
    (trainImages, trainLabels) = gtsrb.getTrainBatch()
    (testImages, testLabels) = gtsrb.getTestBatch()

    (allTestImages, allTestLabels) = (gtsrb.testData, gtsrb.testLabels)
    gtsrb.showImage(allTestImages[50])
