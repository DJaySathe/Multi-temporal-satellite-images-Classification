import pandas as pd
from numpy.random import *
from sklearn.naive_bayes import GaussianNB, MultinomialNB


# Training  and Accuracy on 'ValidationData'

data = pd.read_csv('../TrainingData/ValidationData/ValidationData-2015-04-19.csv')
cols = data.columns

data.drop(cols[0], axis=1, inplace=True)

b = choice(range(len(data)), len(data))

data = data.iloc[b].reset_index()
testIdx = np.random.rand(len(data)) < 0.2

testData = data.iloc[testIdx].reset_index()
trainData = data[~testIdx].reset_index()

cols = trainData.columns
trainData.drop(cols[[0,1]], axis=1, inplace=True)
testData.drop(cols[[0,1]], axis=1, inplace=True)

cols = trainData.columns
counts = trainData['Class'].value_counts()
priorProb = counts/len(trainData)

image1MLCmodel = MultinomialNB(class_prior = priorProb)
image1MLCmodel.fit(trainData[cols[1:]], trainData['Class'])
predictions = image1MLCmodel.predict(testData[cols[1:]])
image1MLCmodel.score(testData[cols[1:]], testData['Class'])


# Accuracy on 'AccuracyTestData'

accuracyTestData = pd.read_csv("../TrainingData/AccuracyTestingData/AccuracyData-2015-12-31.csv")
cols = accuracyTestData.columns
accuracyTestData.drop(cols[[0,1,2]], axis=1, inplace=True)
accuracyTestData = accuracyTestData.reset_index()
cols = accuracyTestData.columns
predictions = image1MLCmodel.predict(accuracyTestData[cols[1:]])
image1MLCmodel.score(accuracyTestData[cols[1:]], accuracyTestData['Class'])
