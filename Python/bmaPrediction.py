import numpy as np
import pandas as pd
import pickle
from MLCFast import *

# load training data not really necessary
dataset1 = pd.read_csv('../Data/Training/ValidationDataImage1.csv')
dataset2 = pd.read_csv('../Data/Training/ValidationDataImage2.csv')
dataset3 = pd.read_csv('../Data/Training/ValidationDataImage3.csv')
dataset4 = pd.read_csv('../Data/Training/ValidationDataImage4.csv')

# load testing data
#dataset1=pd.read_csv("../Data/Testing/AccuracyDataImage1.csv")
#dataset2=pd.read_csv("../Data/Testing/AccuracyDataImage2.csv")
#dataset3=pd.read_csv("../Data/Testing/AccuracyDataImage3.csv")
#dataset4=pd.read_csv("../Data/Testing/AccuracyDataImage4.csv")

cols = dataset1.columns

dataset1.drop(cols[[0,1,2]], inplace=True, axis=1)
dataset2.drop(cols[[0,1,2]], inplace=True, axis=1)
dataset3.drop(cols[[0,1,2]], inplace=True, axis=1)
dataset4.drop(cols[[0,1,2]], inplace=True, axis=1)


with open('../TrainedModels/image1.BMAmodel.LogWeighted.pkl','r') as f:
  model1 = pickle.load(f)
with open('../TrainedModels/image2.BMAmodel.LogWeighted.pkl','r') as f:
  model2 = pickle.load(f)
with open('../TrainedModels/image3.BMAmodel.LogWeighted.pkl','r') as f:
  model3 = pickle.load(f)
with open('../TrainedModels/image4.BMAmodel.LogWeighted.pkl','r') as f:
  model4 = pickle.load(f)


def bmaPrediction(sample):
  if (len(sample) != 4) :
    print("error in input")
    return None

  # sample weights
  #w1 = -120
  #w2 = -70
  #w3 = -200
  #w4 = -90
  w1 = model1.bmaweight
  w2 = model2.bmaweight
  w3 = model3.bmaweight
  w4 = model4.bmaweight


  s = float(w1+w2+w3+w4)
  w1 = w1/s
  w2 = w2/s
  w3 = w3/s
  w4 = w4/s
  w1 = 1 - w1
  w2 = 1 - w2
  w3 = 1 - w3
  w4 = 1 - w4
  
  p1 = pd.DataFrame(model1.predict(sample.iloc[0,1:], type='raw')*w1)
  p2 = pd.DataFrame(model2.predict(sample.iloc[0,1:], type='raw')*w2)
  p3 = pd.DataFrame(model3.predict(sample.iloc[0,1:], type='raw')*w3)
  p4 = pd.DataFrame(model4.predict(sample.iloc[0,1:], type='raw')*w4)
  
  currProbabilities = pd.concat([p1, p2, p3, p4], axis=0).reset_index(drop=True)

  currProbabilities = np.mean(currProbabilities, axis=0)
  c = model1.classes[np.argmax(currProbabilities)]
  return c


if __name__ == '__main__':
  outputDataFrame = dataset1
  predictions = []
  for i in range(len(outputDataFrame)):
    sample = pd.concat([dataset1.iloc[i,], dataset2.iloc[i,], dataset3.iloc[i,], dataset4.iloc[i,]], axis=1).transpose().reset_index(drop=True)
    c = bmaPrediction(sample)
    predictions.append(c)
  print predictions
  accuracy = sum(predictions == outputDataFrame['Class'])/len(predictions)
  print accuracy

# Mean accuracy of 4 models: 0.8958333

#Normal BMA Acc=Trainset
#0.9289617

#Normal BMA Accuracytest set
#0.9111111