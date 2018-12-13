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
        self.unknownStimuliData = {}
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
    lastPulseTime = -1000000
    ecgPrev = currentStimuliData.rawData[0].ecg
    ecgCurr = currentStimuliData.rawData[0].ecg
    timePrev = currentStimuliData.rawData[0].time
    timeCurr = currentStimuliData.rawData[0].time
    spikeSlopeThreshold = 90/10000
    for d in currentStimuliData.rawData:
        ecgCurr = d.ecg
        timeCurr = d.time
        if lastPulseTime < d.time - 500000:
            if(timeCurr-timePrev == 0):
                ecgPrev = ecgCurr
                timePrev = timeCurr
                continue
            slope = (ecgCurr-ecgPrev)/(timeCurr-timePrev)
            if slope > spikeSlopeThreshold:
                if lastPulseTime >= 0:
                    bpmTimes.append(d.time)
                    bpms.append(60000000 / (d.time - lastPulseTime))
                lastPulseTime = d.time
            ecgPrev = ecgCurr
            timePrev = timeCurr
    return (bpmTimes, bpms)

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
    print(xfeatures)
    print(yfeatures)
    print(afeatures)
    print(bfeatures)
    plt.plot(xfeatures, yfeatures, "yo")
    plt.show()
    train_features, test_features, train_labels, test_labels = tts(features, labels, test_size=0.2)
    classify.fit(train_features, train_labels)
    predictions = classify.predict(test_features)
    print(test_features)
    #print(classify.coef_)
    #print("Predictions: ", predictions)
    scores = cross_val_score(classify, features, labels, cv=8)
    #print(scores)
    #print(scores.mean(), scores.std())
    print(predictions)

    """
    print("For KFold \n")

    kf = KFold(n_splits=4, shuffle=True)
    #for train, test in kf.split(features):
        #print(train, test)

    #print("For SKFold \n")

    skf = StratifiedKFold(n_splits=8)
    #for train, test in skf.split(features, labels):
        #print(train, test)

    """

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

def extractSingleFileFeatures(stimuliData):
    features = []
    features.append([stimuliData.normalizedHeartRateRegressionSlope, stimuliData.normalizedGSRRegressionSlope,
                     stimuliData.normalizedHeartRateRegressionYIntercept,
                     stimuliData.normalizedGSRRegressionYIntercept])
    features = np.asarray(features)
    return features

def parseSingleFile(fileName):
    file = open(join(os.getcwd(), fileName), 'r')
    fileName = fileName.replace("/", "\\").split('\\')[len(fileName.split('\\')) - 1]
    fileName = fileName.replace('.txt', '')
    currentParticipant = ParticipantData(fileName)
    for line in file.readlines():
        # insect correspond to negative data, flower correspond to positive data, both used to build model
        # unknown correspond to uncategorized data, used to test using model
        if '.jpg' in line or '#' in line:
            imageName = line.replace('.jpg', '').replace('#', '')
            if ('insect' in imageName):
                currentStimuliData = StimuliData(imageName)
                currentParticipant.negativeStimuliData[imageName] = currentStimuliData
            elif ('flower' in imageName):
                currentStimuliData = StimuliData(imageName)
                currentParticipant.positiveStimuliData[imageName] = currentStimuliData
            else:
                currentStimuliData = StimuliData(imageName)
                currentParticipant.unknownStimuliData[imageName] = currentStimuliData
            startTime = -1

        if len(line.split(',')) == 3:
            if (startTime == -1):
                startTime = int(line.split(',')[0])
            currentStimuliData.rawData.append(DataValue(startTime, line))

    # process data for each stimuli
    for currentStimuliData in set(currentParticipant.positiveStimuliData.values()) | \
                              set(currentParticipant.negativeStimuliData.values()) | \
                              set(currentParticipant.unknownStimuliData.values()):
        currentRawData = currentStimuliData.rawData

        # get data from raw data
        ecgs = []
        gsrs = []
        times = []
        for d in currentRawData:
            ecgs.append(d.ecg)
            gsrs.append(d.gsr)
            times.append(d.time)

        # compute ecg data
        currentStimuliData.avgHeartRate = sum(ecgs) / len(ecgs)
        currentStimuliData.stdDevHeartRate = statistics.stdev(ecgs)
        currentStimuliData.minHeartRate = min(ecgs)
        currentStimuliData.maxHeartRate = max(ecgs)
        (bpmTimes, bpms) = computeBPMs(currentStimuliData)
        # plot data

        (bpmTimes, bpms) = computeBPMs(currentStimuliData)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(times, ecgs)

        plt.subplot(212)
        plt.plot(bpmTimes, bpms)
        plt.title(currentParticipant.name + " " + currentStimuliData.name)
        plt.show()

        # ---call compute bpm function---#
        #bpmTimes, bpms = [0, 1, 2, 3, 4, 5, 6, 7], [300, 400, 500, 600, 500, 300, 200, 100]
        # compute normalized bpms
        normalizedBpms = []
        for bpm in bpms:
            # uses newValue = (value - ave)/std
            normalizedBpms.append((bpm - currentStimuliData.avgHeartRate) / currentStimuliData.stdDevHeartRate)
        currentStimuliData.heartRateTimes = bpmTimes
        currentStimuliData.heartRates = normalizedBpms

        # compute gsr data
        currentStimuliData.avgGSR = sum(gsrs) / len(gsrs)
        currentStimuliData.stdDevGSR = statistics.stdev(gsrs)
        currentStimuliData.minGSR = min(gsrs)
        currentStimuliData.maxGSR = max(gsrs)
        # compute normalized gsrs
        normalizedGSRs = []
        for gsr in gsrs:
            # uses newValue = (value - ave)/std
            if currentStimuliData.stdDevGSR == 0:
                normalizedGSRs.append(0)
            else:
                normalizedGSRs.append((gsr - currentStimuliData.avgGSR) / currentStimuliData.stdDevGSR)

        # add ave heart rate data for each stimuli to participant's data
        currentParticipant.avgHeartRates.append(currentStimuliData.avgHeartRate)
        currentParticipant.stdDevHeartRates.append(currentStimuliData.stdDevHeartRate)

        # add ave gsr data for each stimuli to participant's data
        currentParticipant.avgGSRs.append(currentStimuliData.avgGSR)
        currentParticipant.stdDevGSRs.append(currentParticipant.stdDevGSR)

        # compute lin reg bpm for each stiumli
        heartRateRegSlope, heartRateRegIntercept = np.polyfit(bpmTimes, normalizedBpms, 1)
        currentStimuliData.normalizedHeartRateRegressionSlope = heartRateRegSlope
        currentStimuliData.normalizedHeartRateRegressionYIntercept = heartRateRegIntercept

        # graph lin reg bpm
        """
        plt.plot(np.asarray(bpmTimes), np.asarray(normalizedBpms), 'yo', np.asarray(bpmTimes),
                 heartRateRegSlope * np.asarray(bpmTimes) + heartRateRegIntercept, '--k')
        plt.title(fileName + ": " + imageName)
        plt.xlabel('Time')
        plt.ylabel('BPM')
        plt.show()
        """
        # compute lin reg gsr for each stimuli
        gsrRegSlope, gsrRegIntercept = np.polyfit(times, normalizedGSRs, 1)
        currentStimuliData.normalizedGSRRegressionSlope = gsrRegSlope
        currentStimuliData.normalizedGSRRegressionYIntercept = gsrRegIntercept

        # graph lin reg gsr
        """
        plt.plot(np.asarray(times), np.asarray(normalizedGSRs), 'yo', np.asarray(times),
                 gsrRegSlope * np.asarray(times) + gsrRegIntercept, '--k')
        plt.title(fileName + ": " + imageName)
        plt.xlabel('Time')
        plt.ylabel('GSR')
        plt.show()
        """
    # compute summary data of each participant
    currentParticipant.avgHeartRate = sum(currentParticipant.avgHeartRates) / len(currentParticipant.avgHeartRates)
    currentParticipant.stdDevHeartRate = statistics.stdev(currentParticipant.avgHeartRates)
    currentParticipant.avgGSR = sum(currentParticipant.avgGSRs) / len(currentParticipant.avgGSRs)
    currentParticipant.stdDevGSR = statistics.stdev(currentParticipant.avgGSRs)
    file.close()
    return currentParticipant

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

dataFile = 0
ser = 0
terminateThread = False
threadExecuting = False
dataFileName = ''

class CollectDataThread (threading.Thread):
   def __init__(self):
      threading.Thread.__init__(self)

   def run(self):
       global dataFile
       global ser
       global terminateThread
       global threadExecuting
       while True:
           if ser.in_waiting > 0:
               dataFile.write(ser.readline().decode("utf-8") )
               if (terminateThread):
                   threadExecuting = False
                   return

def initSensors():
    global dataFile
    global ser
    global dataFileName
    ser = serial.Serial(port="COM5", baudrate=19200)
    dataFileName = join("data", "participant","participantData%s.txt" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':','').replace(' ', ''))
    dataFile = open(join(os.getcwd(),dataFileName), "w")
    time.sleep(3)

def stimuliStart(stimuliCategory):
    global dataFile
    global ser
    global terminateThread
    global threadExecuting
    dataFile.write("#")
    dataFile.write(stimuliCategory)
    dataFile.write("\n")
    terminateThread = False
    threadExecuting = True
    collectDataThread = CollectDataThread()
    collectDataThread.start()
    ser.write(b'f')

def stimuliEnd():
    global dataFile
    global ser
    global terminateThread
    ser.write(b'f')
    terminateThread = True
    while threadExecuting:
        pass
    while ser.in_waiting > 0:
        dataFile.write(ser.readline().decode("utf-8") )

    dataFile.write('endOfStimuli\n')

def computeBias():
    global dataFile
    global ser
    global dataFileName
    dataFile.close()
    ser.close()
    participantData = parseSingleFile(dataFileName)

    #build model
    participantsData = parseData("data")
    features, labels = extractFeaturesLabels(participantsData)
    classify = svm.SVC(kernel="rbf")
    classify.fit(features, labels)

    #Test model here
    stimuliCategories = {}

    for stimuliData in participantData.unknownStimuliData.values():
        singleFeatures = extractSingleFileFeatures(stimuliData)
        result = classify.predict(singleFeatures)
        if(result[0] == 0):
            stimuliCategories[stimuliData.name] = True
        else:
            stimuliCategories[stimuliData.name] = False
    ################
    return stimuliCategories
    #return a map of "stimuliCategory"-> bias:boolean



#participantsData = parseData("data")

#buildModel(participantsData)

initSensors()

stimuliStart("Category 1")
time.sleep(3)
stimuliEnd()

stimuliStart("Category 2")
time.sleep(3)
stimuliEnd()

categories = computeBias()

for key in categories.keys():
    print(key, categories[key])

#below this is dead code.  please disregard
'''
file = open('calibrationDataEric1.txt', 'r')
allData = {}
imageName = ''
for line in file.readlines():
    if '.jpg' in line:
        imageName = line
        allData[imageName] = []
        startTime = -1
    else:
        if len(line.split(',')) == 3:
            if(startTime ==  -1):
                startTime = int(line.split(',')[0])
            allData[imageName].append(DataValue(startTime, line))

for image in allData:
    data = allData[image]
    ecgs = [d.ecg for d in data]
    avgECG = sum(ecgs)/len(ecgs)
    stdDevECG = statistics.stdev(ecgs)
    bpmTimes = []
    bpms = []
    lastPulseTime = -1000
    for d in data:
        if lastPulseTime < d.time - 150000:
            if d.ecg > 600:
                if lastPulseTime >= 0:
                    bpmTimes.append(d.time)
                    bpms.append(60000000/(d.time - lastPulseTime))
                lastPulseTime = d.time

    times = [d.time for d in data]
    ecgs = [d.ecg for d in data]

    plt.figure(1)
    plt.subplot(211)
    plt.plot(times, ecgs)

    plt.subplot(212)
    plt.plot(bpmTimes, bpms)
    plt.title(image)
    plt.show()
'''

#possibly dead code if we're using dic and not sets
"""
    for currentStimuliData in set(currentParticipant.positiveStimuliData) | set(currentParticipant.negativeStimuliData):
        ecgs = [d.ecg for d in currentStimuliData.rawData]
        gsrs = [d.ecg for d in currentStimuliData.rawData]
        currentStimuliData.avgHeartRate = sum(ecgs) / len(ecgs)
        currentStimuliData.stdDevHeartRate = statistics.stdev(ecgs)
        currentStimuliData.avgGSR = sum(gsrs) / len(gsrs)
        currentStimuliData.stdDevGSR = statistics.stdev(gsrs)
        (bpmTimes, bpms) = computeBPMs(currentStimuliData)
        currentStimuliData.heartRateTimes = bpmTimes
        currentStimuliData.heartRates = bpms
        currentParticipant.avgHeartRates.append(currentStimuliData.avgHeartRate)
        currentParticipant.stdDevHeartRates.append(currentStimuliData.stdDevHeartRate)
        currentParticipant.gsrs.append(gsrs)
"""
