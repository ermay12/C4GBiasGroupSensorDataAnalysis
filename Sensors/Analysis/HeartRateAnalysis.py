import matplotlib.pyplot as plt
import statistics
from os import listdir, getcwd
from os.path import isfile, join


class DataValue:
    def __init__(self,startTime, line):
        self.time = int(line.split(',')[0]) - startTime
        self.ecg = int(line.split(',')[1])
        self.gsr = int(line.split(',')[2])

class StimuliData:
    def __init__(self, imageName):
        self.name = imageName
        self.ecgData = []
        self.heartRates = []
        self.avgHeartRate = 0.0
        self.stdDevHeartRate = 0.0
        self.minHeartRate = 0.0
        self.maxHeartRate = 0.0
        self.normalizedHeartRateRegressionSlope = 0.0
        self.normalizedHeartRateRegressionYIntercept = 0.0
        self.avgGSR = 0.0
        self.stdDevGSR = 0.0
        self.minGSR = 0.0
        self.maxGSR = 0.0
        self.normalizedGSRRegressionSlope = 0.0
        self.normalizedGSRRegressionYIntercept = 0.0

class ParticipantData:
    def __init__(self, name):
        self.name = name
        self.positiveStimuliData = {}
        self.negativeStimuliData = {}
        self.avgHeartRate = 0.0
        self.stdDevHeartRate = 0.0
        self.heartRates = []
        self.avgGSR = 0.0
        self.stdDevGSR = 0.0

def processRawData(dataDirectory):
    dataDirectory = join(getcwd(), dataDirectory)
    filePaths = [join(dataDirectory, f) for f in listdir(dataDirectory) if isfile(join(dataDirectory, f))]
    participantsData = {}
    for filePath in filePaths:
        if('.txt' not in fileName):
            continue
        fileName = filePath.split('\\')[len(filePath.split('\\'))-1]
        fileName = fileName.replace('.txt', '')
        currentParticipant = ParticipantData(fileName)
        participantsData[fileName] = currentParticipant
        file = open(filePath, 'r')
        imageName = ''
        for line in file.readlines():
            if '.jpg' in line or '#' in line:
                imageName = line.replace('.jpg', '').replace('#', '')
                if('insect' in imageName):
                    currentStimuliData = StimuliData(imageName)
                    currentParticipant.negativeStimuliData.append(currentStimuliData)
                else:
                    currentStimuliData = StimuliData(imageName)
                    currentParticipant.positiveStimuliData.append(currentStimuliData)
                startTime = -1
            else:
                if len(line.split(',')) == 3:
                    if (startTime == -1):
                        startTime = int(line.split(',')[0])
                    currentStimuliData.ecgData.append(DataValue(startTime, line))

        ecgs = [d.ecg for d in currentStimuliData.ecgData]
        gcrs = [d.ecg for d in currentStimuliData.ecgData]
        currentStimuliData.avgHeartRate = sum(ecgs) / len(ecgs)
        currentStimuliData.stdDevHeartRate = statistics.stdev(ecgs)
        currentStimuliData.avgGSR = sum(ecgs) / len(ecgs)
        currentStimuliData.stdDevGSR = statistics.stdev(ecgs)
        bpmTimes = []
        bpms = []
        lastPulseTime = -1000
        for d in data:
            if lastPulseTime < d.time - 150000:
                if d.ecg > 600:
                    if lastPulseTime >= 0:
                        bpmTimes.append(d.time)
                        bpms.append(60000000 / (d.time - lastPulseTime))
                    lastPulseTime = d.time

        times = [d.time for d in data]
        ecgs = [d.ecg for d in data]


processRawData('data')

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
