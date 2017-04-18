import pandas as pd
from numpy.random import *
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB


# Training  and Accuracy on 'ValidationData'

data = pd.read_csv('../Data/Training/ValidationDataImage4.csv')
cols = data.columns

data.drop(cols[0], axis=1, inplace=True)
cols = data.columns
b = choice(range(len(data)), len(data))

data = data.iloc[b].reset_index(drop=True)
testIdx = np.random.rand(len(data)) < 0.2

testData = data.iloc[testIdx].reset_index(drop=True)
trainData = data[~testIdx].reset_index(drop=True)

cols = trainData.columns
trainData.drop(cols[[0,1]], axis=1, inplace=True)
testData.drop(cols[[0,1]], axis=1, inplace=True)

cols = trainData.columns
counts = trainData['Class'].value_counts(sort=False)
priorProb = counts/len(trainData)

image1MLCmodel = GaussianNB()
X = trainData[cols[1:]]
y = trainData['Class']
image1MLCmodel.fit(X, y)
predictions = image1MLCmodel.predict(testData[cols[1:]])
print image1MLCmodel.score(testData[cols[1:]], testData['Class'])


# Accuracy on 'AccuracyTestData'

accuracyTestData = pd.read_csv("../Data/Testing/AccuracyDataImage4.csv")
cols = accuracyTestData.columns
accuracyTestData.drop(cols[[0,1,2]], axis=1, inplace=True)
accuracyTestData = accuracyTestData.reset_index(drop=True)
cols = accuracyTestData.columns
predictions = image1MLCmodel.predict(accuracyTestData[cols[1:]])
print image1MLCmodel.score(accuracyTestData[cols[1:]], accuracyTestData['Class'])
