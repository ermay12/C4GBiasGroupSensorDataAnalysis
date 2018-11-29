import matplotlib.pyplot as plt
import statistics
from os import listdir, getcwd
from os.path import isfile, join

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


### Jade redo this please! make it more robust so that it works for everyone###
def computeBPMs(currentStimuliData):
    bpmTimes = []
    bpms = []
    lastPulseTime = -1000
    for d in currentStimuliData.ecgData:
        if lastPulseTime < d.time - 150000:
            if d.ecg > 600:
                if lastPulseTime >= 0:
                    bpmTimes.append(d.time)
                    bpms.append(60000000 / (d.time - lastPulseTime))
                    lastPulseTime = d.time
    return (bpmTimes, bpms)

#processes all the data in the dataDirectory folder and returns a list of ParticipantData
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
                    currentStimuliData.rawData.append(DataValue(startTime, line))

        for currentStimuliData in set(currentParticipant.positiveStimuliData) | set(currentParticipant.negativeStimuliData):
            ecgs = [d.ecg for d in currentStimuliData.ecgData]
            gsrs = [d.ecg for d in currentStimuliData.ecgData]
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
            ###Jade fill in the rest of the currentStimuli's data here###
            ###Disregard the normalized regressions for now

            ###-------------------------------------------------------###
        currentParticipant.avgHeartRate = sum(currentParticipant.avgHeartRates)/len(currentParticipant.avgHeartRates)
        currentParticipant.stdDevHeartRate = -1#Jade figure this out.  lol i forget std deviation
        currentParticipant.avgGSR = sum(currentParticipant.gsrs) / len(currentParticipant.gsrs)
        currentParticipant.stdDevGSR = statistics.stdev(currentParticipant.gsrs)
        ### Jade  compute the normalized linear regressions for each stimuli         ###

        ###--------------------------------------------------------------------------###
    return participantsData


participantsData = processRawData('data')

### Jade graph everything here ###


###----------------------------###




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
