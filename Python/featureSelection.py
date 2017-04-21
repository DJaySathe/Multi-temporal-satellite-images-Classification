import pandas as pd
import numpy as np
import pickle
import itertools as it
import operator
from MLCFast import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def bestSubsetSelection():
  imageNumber = 4
  data = pd.read_csv('../Data/Training/ValidationDataImage'+str(imageNumber)+'.csv')
  cols = data.columns
  data.drop(cols[[0,1,2]], inplace=True, axis=1)

  X = data.iloc[:,1:]
  cols = X.columns

  best_cols = {}

  for r in range(1, len(cols)+1):
    x = it.combinations(range(len(cols)), r)
    combs = list(x)
    for c in combs:
      cor = cols[list(c)]
      accuracy = runIteration(cor, imageNumber)
      best_cols[c] = accuracy

  return best_cols

def dimReductionPCA(imageNumber):
	data = pd.read_csv('../Data/Training/ValidationDataImage'+str(imageNumber)+'.csv')
	cols = data.columns
	data.drop(cols[[0,1,2]], inplace=True, axis=1)

	X = data.iloc[:,1:]
	cols = X.columns

	pca = PCA()

	scaler = StandardScaler()
	scaler.fit(X)

	pca.fit(scaler.transform(X))

	pcaX = pca.transform(scaler.transform(X))

	pcaX = pd.DataFrame(pcaX)

	output = pd.concat([data['Class'], pcaX.iloc[:,:5]], axis=1)
	output.to_csv('../Data/Training/pcaImage'+str(imageNumber)+'.csv', index=False)

	data2 = pd.read_csv('../Data/Testing/AccuracyDataImage'+str(imageNumber)+'.csv')
	cols = data2.columns
	data2.drop(cols[[0,1,2]], inplace=True, axis=1)

	testX = data2.iloc[:,1:]
	pcaTestX = pca.transform(scaler.transform(testX))

	pcaTestX = pd.DataFrame(pcaTestX)
	output2 = pd.concat([data2['Class'], pcaTestX.iloc[:,:5]], axis=1)
	output2.to_csv('../Data/Testing/pcaTestImage'+str(imageNumber)+'.csv', index=False)
	#pcaX.iloc[:,:5].to_csv('../Data/Training/pcaImage'+str(imageNumber)+'.csv', index=False)
	#print pca.explained_variance_ratio_
	#print pca.explained_variance_ratio_[0:5].sum()


def runIteration(cols, imageNumber):
  data = pd.read_csv('../Data/Training/ValidationDataImage'+str(imageNumber)+'.csv')
  
  X = data[cols]
  y = data['Class']

  accuracyTestData = pd.read_csv('../Data/Testing/AccuracyDataImage'+str(imageNumber)+'.csv')
  
  Xtest = accuracyTestData[cols]
  ytest = accuracyTestData['Class']

  model = MLCFast()
  model.fit(X, y)

  preds = model.predict(Xtest)
  accuracy = model.score(preds, ytest)
  return accuracy


def main1():
	d = bestSubsetSelection()
	sd = sorted(d.items(), key=operator.itemgetter(1))
	print sd

def main2():
	dimReductionPCA(1)
	dimReductionPCA(2)
	dimReductionPCA(3)
	dimReductionPCA(4)

if __name__ == '__main__':
	main2()