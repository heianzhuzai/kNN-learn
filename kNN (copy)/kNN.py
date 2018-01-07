from numpy import *
import operator
from os import listdir

def createDateStr():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def img2vector(filename):
	returnVect = zeros((1, 1024))

	fr = open(filename)

	for i in range(32):
		lineStr = fr.readline()

		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])

	return returnVect

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet

	aqDiffMat = diffMat**2
	sqDistance = aqDiffMat.sum(axis=1)

	distances = sqDistance**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}

	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)

	return sortedClassCount[0][0]

def handwritingClassTest():
	hwLables = []
	errorR= []
	trainingFileList = listdir('digits/trainingDigits')
	m = len(trainingFileList)

	trainingMat = zeros((m, 1024))

	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLables.append(classNumStr)

		trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)

	testFileList = listdir('digits/testDigits')
	errorCount = 0.0
	mTest = len(testFileList)

	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])

		vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLables, 5)
		print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
		errorData = "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
		if (classifierResult != classNumStr):
			errorCount += 1.0
			errorR.append(errorData)

	print "\nthe total number of errors is: %d" % errorCount
	print "\nthe tatal error rate is: %f" % (errorCount/float(mTest))
	for i in range(len(errorR)):
		print errorR[i]

handwritingClassTest()


