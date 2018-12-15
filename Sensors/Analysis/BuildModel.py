import matplotlib.pyplot as plt
import statistics
import os
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as tts
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import array as arr
import datetime
import threading
import serial
import time
import pickle
from MeasureBias import ParticipantData, StimuliData, DataValue, extractFeaturesLabels, \
    extractSingleFileFeatures, parseSingleFile

#build model using only data categorized as positive or negative stimuli data
def buildModel(participantsData):
    classify = svm.SVC(kernel="rbf")

    #extract features and labels from data
    features = []
    labels = []
    pKeys = participantsData.keys()
    for pKey in pKeys:
        participant = participantsData[pKey]
        positive = participant.positiveStimuliData
        negative = participant.negativeStimuliData
        pSKeys = positive.keys()
        nSKeys = negative.keys()
        for pSKey in pSKeys:
            imageData = positive[pSKey]
            features.append([imageData.normalizedHeartRateRegressionSlope, imageData.normalizedGSRRegressionSlope,
                            imageData.normalizedHeartRateRegressionYIntercept, imageData.normalizedGSRRegressionYIntercept])
            labels.append(1)
        for nSKey in nSKeys:
            imageData = negative[nSKey]
            features.append([imageData.normalizedHeartRateRegressionSlope, imageData.normalizedGSRRegressionSlope,
                             imageData.normalizedHeartRateRegressionYIntercept, imageData.normalizedGSRRegressionYIntercept])
            labels.append(0)

    #build model
    print(len(features))
    features = np.asarray(features)
    labels = np.asarray(labels)
    xfeatures = []
    yfeatures = []
    afeatures = []
    bfeatures = []
    for feature in features:
        xfeatures.append(feature[0])
        yfeatures.append(feature[1])
        afeatures.append(feature[2])
        bfeatures.append(feature[3])
    """
    print(xfeatures)
    print(yfeatures)
    print(afeatures)
    print(bfeatures)
    plt.plot(xfeatures, yfeatures, "yo")
    plt.show()
    """
    train_features, test_features, train_labels, test_labels = tts(features, labels)
    classify.fit(train_features, train_labels)

    return classify

def extractFeaturesLabels(participantsData):
    features = []
    labels = []
    pKeys = participantsData.keys()
    for pKey in pKeys:
        participant = participantsData[pKey]
        positive = participant.positiveStimuliData
        negative = participant.negativeStimuliData
        pSKeys = positive.keys()
        nSKeys = negative.keys()
        for pSKey in pSKeys:
            imageData = positive[pSKey]
            features.append([imageData.normalizedHeartRateRegressionSlope, imageData.normalizedGSRRegressionSlope,
                             imageData.normalizedHeartRateRegressionYIntercept,
                             imageData.normalizedGSRRegressionYIntercept])
            labels.append(1)
        for nSKey in nSKeys:
            imageData = negative[nSKey]
            features.append([imageData.normalizedHeartRateRegressionSlope, imageData.normalizedGSRRegressionSlope,
                             imageData.normalizedHeartRateRegressionYIntercept,
                             imageData.normalizedGSRRegressionYIntercept])
            labels.append(0)

    # build model
    features = np.asarray(features)
    labels = np.asarray(labels)
    return features, labels

#processes all the data in the dataDirectory folder and returns a list of ParticipantData
def parseData(dataDirectory):
    dataDirectory = join(os.getcwd(), dataDirectory)
    participantsData = {}
    for fileName in os.listdir(dataDirectory):
        if '.txt' not in fileName or 'participant' in fileName:
            continue
        currentParticipant = parseSingleFile(join('data',fileName))
        participantsData[fileName] = currentParticipant


    return participantsData



participantsData = parseData("data")

model = buildModel(participantsData)

with open("model.pkl", 'wb') as output:  # Overwrites any existing file.
    pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
