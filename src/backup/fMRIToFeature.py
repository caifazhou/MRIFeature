import os, glob
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import nibabel as nib
from scipy import stats
from nibabel.testing import data_path

# -------------------------------------------------
# functions
# csv write a formated csv file
def csvFormatedOutput(fileName, ans):
    with open(fileName, 'w', newline='') as f:
        c = csv.writer(f, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(['ID', 'Prediction'])
        for i in range(0, len(ans)):
            c.writerow([i + 1, ans[i], ])
    return 0

def csvMLP3FormatedOutput(fileName, ans):
    label = []
    label.append('gender')
    label.append('age')
    label.append('health')

    with open(fileName, 'w', newline='') as f:
        c = csv.writer(f, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(['ID', 'Sample', 'Label', 'Predicted'])
        for i in range(0, len(ans)):
            for j in range(len(ans[i])):
                c.writerow([3*i + j, i, label[j], ans[i]])
    return 0

def csvOutput(fileName, ans):
    with open(fileName, 'w', newline='') as f:
        c = csv.writer(f, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # c.writerow(['ID', 'Prediction'])
        for i in range(0, len(ans)):
            c.writerow(ans[i])
    return 0

def listToMat(l):
    return np.array(l).reshape(-1, np.array(l).shape[-1])


def plotter(measured, predicted):
    # plot the result
    fig, ax = plt.subplots()
    y = measured
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    return 0


def readFileDir(fileDir):
    dirFile = []
    for file in os.listdir(fileDir):
        if file.endswith(".nii"):
            dirFile.append(file)
    dirFile = listToMat(dirFile)
    numDir = dirFile.shape[1]
    if numDir == 0:
        return [], -1
    else:
        return dirFile, numDir


def readFileNumber(filename):
    startIndex = filename.find('_')
    endIndex = filename.find('.')
    if (startIndex != -1) & (endIndex > startIndex):
        return int(filename[startIndex + 1:endIndex])
    else:
        return -1


def computeStats(yy):
    temp = []
    temp.append(yy.mean())
    temp.append(yy.var())
    temp.append(yy.std())
    temp.append(np.median(yy))
    temp.append(stats.skew(yy))
    temp.append(stats.kurtosis(yy))
    temp.append(stats.moment(yy, 3))
    temp.append(stats.moment(yy, 4))
    # temp.append(computeHist(yy))
    return (temp)

def computeHistR(yy, bins):
    temp = []
    xx = np.histogram(yy, bins=bins, density=True)
    temp.append(xx[0])
    return (np.reshape(temp,[-1, ]))

def computeHist(yy, bins):
    temp = []
    xx = np.histogram(yy, bins=bins, density=False)
    temp.append(xx[0])
    return (np.reshape(temp,[-1, ]))

def halfSplit(data):
    x1 = np.vsplit(data, 2)
    x2 = []
    x3 = []
    for i in range(len(x1)):
        temp = (np.hsplit(x1[i], 2))
        for j in range(len(temp)):
            x2.append(temp[j])
    for j in range(len(x2)):
        temp = (np.dsplit(x2[j], 2))
        for j in range(len(temp)):
            x3.append(temp[j])
    return x3

def splitter3d(data, block):
    x1 = np.vsplit(data, block)
    x2 = []
    x3 = []
    for i in range(len(x1)):
        temp = (np.hsplit(x1[i], block))
        for j in range(len(temp)):
            x2.append(temp[j])
    for j in range(len(x2)):
        temp = (np.dsplit(x2[j], block))
        for j in range(len(temp)):
            x3.append(temp[j])
    return x3

def blockDivision(data, level):
    if level == 0:
        return (data)
    if level == 1:
        return halfSplit(data)
    if level == 2:
        temp = halfSplit(data)
        output = []
        for i in range(len(temp)):
            temp1 = halfSplit(temp[i])
            for j in range(len(temp1)):
                output.append(temp1[j])
        return output
    if level == 3:
        temp = halfSplit(data)
        output = []
        for i in range(len(temp)):
            temp1 = halfSplit(temp[i])
            for j in range(len(temp1)):
                temp2 = halfSplit(temp1[j])
                for k in range(len(temp2)):
                    output.append(temp2[k])
        return output

def allBlock(data):
    x0 = blockDivision(data, 0)
    x1 = blockDivision(data, 1)
    x2 = blockDivision(data, 2)
    x3 = blockDivision(data, 3)
    totalBlocks = []
    totalBlocks.append(x0)
    for i in range(len(x1)):
        totalBlocks.append(x1[i])
    for i in range(len(x2)):
        totalBlocks.append(x2[i])
    for i in range(len(x3)):
        totalBlocks.append(x3[i])
    return totalBlocks


def statsFeature(data):
    features = []
    total = allBlock(data)
    for i in range(len(total)):
        tempFeature = computeStats(np.reshape(total[i], [-1, ]))
        for j in range(len(tempFeature)):
            features.append(tempFeature[j])
    return features

def histFeature(data, block, bins):
    features = []
    total = splitter3d(data, block)
    for i in range(len(total)):
        tempFeature = computeHist(np.reshape(total[i], [-1, ]), bins)
        for j in range(len(tempFeature)):
            features.append(tempFeature[j])
    return features

def mriToStatsFeature(fileDir):
    file, number = readFileDir(fileDir)
    numberStats = 8
    numberBlock = 585
    features = np.zeros([number, numberStats*numberBlock])
    # get the MRI image
    for i in range(number):
        filename = file[0,i]
        mriNumber = readFileNumber(filename)
        img = nib.load(fileDir + filename)
        imgData = img.get_data()
        d2 = np.reshape(imgData, [imgData.shape[0],imgData.shape[1],imgData.shape[2]])
        features[mriNumber - 1,:] = statsFeature(d2)
    return features

def mriToHistFeature(fileDir, numberBins, block):
    file, number = readFileDir(fileDir)
    # numberBins = 64
    numberBlock = block*block*block
    features = np.zeros([number, numberBins*numberBlock])
    # get the MRI imageL
    for i in range(number):
        filename = file[0,i]
        mriNumber = readFileNumber(filename)
        img = nib.load(fileDir + filename)
        imgData = img.get_data()
        d2 = np.reshape(imgData, [imgData.shape[0],imgData.shape[1],imgData.shape[2]])
        features[mriNumber - 1,:] = histFeature(d2, block, numberBins)
    return features
# ---------------------------------------

# set the directories
testDir = '../testData/testOrg/set_test/'
trainDir = '../testData/testOrg/set_train/'

numberSeg = 200
testSegDir = '../testData/testSeg/set_test_seg/'
trainSegDir = '../testData/testSeg/set_train_seg/'

fileNames, numberMRIs = readFileDir(trainDir)
for i in range(numberMRIs):
    fileName = fileNames[0, i]
    mriNum = readFileNumber(fileName)
#     mri data
    mriData = np.reshape(nib.load(trainDir + fileName).get_data(), [-1, ])
    # print(nib.load(trainDir + fileName).get_data().shape)
    mriSeg = np.reshape(nib.load(trainSegDir + fileName).get_data(), [-1, ])
    print(mriSeg[21000:21999])

    # for j in range(1,)
# featureTest = mriToStatsFeature(testDir)
# csvOutput('../testFeatureNew1.csv', featureTest)
# featureTrain = mriToStatsFeature(trainDir)
# csvOutput('../trainFeatureNew1.csv', featureTrain)

# histogram feature
# bins = 64
# block = 8
# testDir = '../data/set_test/'
# trainDir = '../data/set_train/'
#
# featureTest = mriToHistFeature(testDir, bins, block)
# csvOutput('../testFeatureHistR.csv', featureTest)
# featureTrain = mriToHistFeature(trainDir, bins, block)
# csvOutput('../trainFeatureHistR.csv', featureTrain)

