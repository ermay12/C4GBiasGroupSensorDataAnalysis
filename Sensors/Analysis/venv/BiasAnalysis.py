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

# represents a single data point read from the arduino.  (time (ns), ecg (raw), gsr (raw))
# time starts at 0 at the beginning of the image.
class DataValue:
    def __init__(self,startTime, line):
        self.time = int(line.split(',')[0]) - startTime
        self.ecg = int(line.split(',')[1])
        self.gsr = int(line.split(',')[2])

# represents all the data associated with an image
class StimuliData:
    def __init__(self, imageName):
        # image name not including the '.jpg'
        self.name = imageName
        # the raw data from the arduino (list of DataValues)
        self.rawData = []
        # the computed heart rates per heart beat
        self.heartRates = []
        # time of each heart beat
        self.heartRateTimes = []
        # avg heart rate for this image
        self.avgHeartRate = 0.0
        # std deviation for the heart rate of this image
        self.stdDevHeartRate = 0.0
        # the min heart rate for this image
        self.minHeartRate = 0.0
        # the max heart rate for this image
        self.maxHeartRate = 0.0
        # the slope and y-intercept of the line of best fit for the graph of
        # heart beats after being normalized for the persons avg and stdDev
        # heart rate.  Must be computed after all images are analyzed
        self.normalizedHeartRateRegressionSlope = 0.0
        self.normalizedHeartRateRegressionYIntercept = 0.0
        self.heartRateFunction = 0.0
        # avg gsr reading
        self.avgGSR = 0.0
        # std deviation of the gsr's values
        self.stdDevGSR = 0.0
        self.minGSR = 0.0
        self.maxGSR = 0.0
        # same as the heart beat's linear regression
        self.normalizedGSRRegressionSlope = 0.0
        self.normalizedGSRRegressionYIntercept = 0.0


# represents the data associated with each participant
class ParticipantData:
    def __init__(self, name):
        # name of the text file this data came from. excludes '.txt'
        self.name = name
        # map of image name to the StimuliData associated with that image
        self.positiveStimuliData = {}
        self.negativeStimuliData = {}
        # average heartrate of all images combined
        self.avgHeartRate = 0.0
        self.stdDevHeartRate = 0.0
        # average gsr of all images combined
        self.avgGSR = 0.0
        self.stdDevGSR = 0.0

        # all average heart rates compiled here to make calculating the average rate easier
        self.avgHeartRates = []
        self.stdDevHeartRates = []

        # all gsr reading for the participant
        self.gsrs = []
        self.avgGSRs = []
        self.stdDevGSRs = []


def computeBPMs(currentStimuliData):
    bpmTimes = []
    bpms = []
    lastPulseTime = -1000
    ecgPrev = currentStimuliData.rawData[0]
    ecgCurr = currentStimuliData.rawData[0]
    spikeSlope = 0
    for d in currentStimuliData.rawData:
        if lastPulseTime < d.time - 150000:
            average = (currentStimuliData.minHeartRate + currentStimuliData.maxHeartRate)/2
            if d.ecg > average:
                #print("ok")
                if lastPulseTime >= 0:
                    #print("Okkkkr")
                    bpmTimes.append(d.time)
                    bpms.append(60000000 / (d.time - lastPulseTime))
                    lastPulseTime = d.time
    return (bpmTimes, bpms)

def analyzeData(participantData):
    print("Default CV")

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
            features.append([imageData.normalizedHeartRateRegressionSlope, imageData.normalizedGSRRegressionSlope, 10, 10])
            labels.append(1)
        for nSKey in nSKeys:
            imageData = negative[nSKey]
            features.append([imageData.normalizedHeartRateRegressionSlope, imageData.normalizedGSRRegressionSlope, 5, 10])
            labels.append(0)

    #build model
    print(len(features))
    features = np.asarray(features)
    labels = np.asarray(labels)
    xfeatures = []
    yfeatures = []
    for feature in features:
        xfeatures.append(feature[0])
        yfeatures.append(feature[1])
    #print(xfeatures)
    #print(yfeatures)
    plt.plot(xfeatures, yfeatures, "yo")
    plt.show()
    train_features, test_features, train_labels, test_labels = tts(features, labels, test_size=0.2)
    classify.fit(train_features, train_labels)
    predictions = classify.predict(test_features)
    #print(classify.coef_)
    #print("Predictions: ", predictions)
    scores = cross_val_score(classify, features, labels, cv=8)
    #print(scores)
    #print(scores.mean(), scores.std())

#processes all the data in the dataDirectory folder and returns a list of ParticipantData
def processRawData(dataDirectory):
    dataDirectory = join(os.getcwd(), dataDirectory)
    participantsData = {}
    for fileName in os.listdir(dataDirectory):
        if('.txt' not in fileName):
            continue
        file = open(dataDirectory + "/" + fileName, 'r')
        fileName = fileName.split('\\')[len(fileName.split('\\'))-1]
        fileName = fileName.replace('.txt', '')
        currentParticipant = ParticipantData(fileName)
        participantsData[fileName] = currentParticipant
        imageName = ''
        for line in file.readlines():
            if '.jpg' in line or '#' in line:
                imageName = line.replace('.jpg', '').replace('#', '')
                if('insect' in imageName):
                    currentStimuliData = StimuliData(imageName)
                    currentParticipant.negativeStimuliData[imageName] = currentStimuliData
                else:
                    currentStimuliData = StimuliData(imageName)
                    currentParticipant.positiveStimuliData[imageName] = currentStimuliData
                startTime = -1

            if len(line.split(',')) == 3:
                if (startTime == -1):
                    startTime = int(line.split(',')[0])
                currentStimuliData.rawData.append(DataValue(startTime, line))

    #code with dic
        for currentStimuliData in set(currentParticipant.positiveStimuliData.values()) | set(currentParticipant.negativeStimuliData.values()):
            currentRawData = currentStimuliData.rawData
            ecgs = []
            gsrs = []
            times = []
            for d in currentRawData:
                ecgs.append(d.ecg)
                gsrs.append(d.gsr)
                times.append(d.time)

            #compute ecg data
            currentStimuliData.avgHeartRate = sum(ecgs) / len(ecgs)
            currentStimuliData.stdDevHeartRate = statistics.stdev(ecgs)
            currentStimuliData.minHeartRate = min(ecgs)
            currentStimuliData.maxHeartRate = max(ecgs)

            (bpmTimes, bpms) = computeBPMs(currentStimuliData)
            plt.figure(1)
            plt.subplot(211)
            plt.plot(times, ecgs)

            plt.subplot(212)
            plt.plot(bpmTimes, bpms)
            plt.title(currentParticipant.name + " " + currentStimuliData.name)
            plt.show()
            bpmTimes, bpms = [0, 1, 2, 3, 4, 5, 6, 7], [300, 400, 500, 600, 500, 300, 200, 100]
            normalizedBpms = []
            #maxBPM = max(bpms)
            #minBPM = min(bpms)
            for bpm in bpms:
                #normalizedBpms.append((bpm - minBPM)/(maxBPM - minBPM))
                normalizedBpms.append((bpm - currentStimuliData.avgHeartRate)/currentStimuliData.stdDevHeartRate)

            currentStimuliData.heartRateTimes = bpmTimes
            currentStimuliData.heartRates = normalizedBpms

            #compute gsr data
            currentStimuliData.avgGSR = sum(gsrs) / len(gsrs)
            currentStimuliData.stdDevGSR = statistics.stdev(gsrs)
            currentStimuliData.minGSR = min(gsrs)
            currentStimuliData.maxGSR = max(gsrs)
            #minGSRS = min(gsrs)
            #maxGSRS = max(gsrs)
            normalizedGSRs = []
            for gsr in gsrs:
                #normalizedGSRs.append((gsr - minGSRS)/(maxGSRS - minGSRS))
                normalizedGSRs.append((gsr - currentStimuliData.avgGSR)/currentStimuliData.stdDevGSR)

            #compute ave heart rate data
            currentParticipant.avgHeartRates.append(currentStimuliData.avgHeartRate)
            currentParticipant.stdDevHeartRates.append(currentStimuliData.stdDevHeartRate)

            #compute ave gsr data
            currentParticipant.avgGSRs.append(currentStimuliData.avgGSR)
            currentParticipant.stdDevGSRs.append(currentParticipant.stdDevGSR)

            #compute lin reg bpm
            heartRateRegSlope, heartRateRegIntercept = np.polyfit(bpmTimes, normalizedBpms, 1)
            currentStimuliData.normalizedHeartRateRegressionSlope = heartRateRegSlope
            currentStimuliData.normalizedHeartRateRegressionYIntercept = heartRateRegIntercept

            #graph lin reg bpm

            """
            plt.plot(np.asarray(bpmTimes), np.asarray(normalizedBpms), 'yo', np.asarray(bpmTimes),
                     heartRateRegSlope * np.asarray(bpmTimes) + heartRateRegIntercept, '--k')
            plt.title(fileName + ": " + imageName)
            plt.xlabel('Time')
            plt.ylabel('BPM')
            plt.show()

            """
            #compute lin reg gsr
            gsrRegSlope, gsrRegIntercept = np.polyfit(times, normalizedGSRs, 1)
            currentStimuliData.normalizedGSRRegressionSlope = gsrRegSlope
            currentStimuliData.normalizedGSRRegressionSlope = gsrRegIntercept

            #graph lin reg gsr

            """
            plt.plot(np.asarray(times), np.asarray(normalizedGSRs), 'yo', np.asarray(times),
                     gsrRegSlope * np.asarray(times) + gsrRegIntercept, '--k')
            plt.title(fileName + ": " + imageName)
            plt.xlabel('Time')
            plt.ylabel('GSR')
            plt.show()

            """
        #compute summary data of each participant
        currentParticipant.avgHeartRate = sum(currentParticipant.avgHeartRates)/len(currentParticipant.avgHeartRates)
        currentParticipant.stdDevHeartRate = statistics.stdev(currentParticipant.avgHeartRates)
        currentParticipant.avgGSR = sum(currentParticipant.avgGSRs) / len(currentParticipant.avgGSRs)
        currentParticipant.stdDevGSR = statistics.stdev(currentParticipant.avgGSRs)

    return participantsData

participantsData = processRawData("data")

analyzeData(participantsData)




