import pandas as pd
import numpy as np
import pickle
import itertools as it
import operator
from MLCFast import *


def bestSubsetSelection():
  imageNumber = 1
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


if __name__ == '__main__':
	d = bestSubsetSelection()
	sd = sorted(d.items(), key=operator.itemgetter(1))
	print sd