import numpy as np
import pandas as pd
import pickle
from MLCFast import *
import statsmodels.api as sm

# load training data not really necessary
# dataset1 = pd.read_csv('../Data/Training/ValidationDataImage1.csv')
# dataset2 = pd.read_csv('../Data/Training/ValidationDataImage2.csv')
# dataset3 = pd.read_csv('../Data/Training/ValidationDataImage3.csv')
# dataset4 = pd.read_csv('../Data/Training/ValidationDataImage4.csv')

# load testing data
dataset1=pd.read_csv("../Data/Testing/AccuracyDataImage1.csv")
dataset2=pd.read_csv("../Data/Testing/AccuracyDataImage2.csv")
dataset3=pd.read_csv("../Data/Testing/AccuracyDataImage3.csv")
dataset4=pd.read_csv("../Data/Testing/AccuracyDataImage4.csv")

cols = dataset1.columns

dataset1.drop(cols[[0,1,2]], inplace=True, axis=1)
dataset2.drop(cols[[0,1,2]], inplace=True, axis=1)
dataset3.drop(cols[[0,1,2]], inplace=True, axis=1)
dataset4.drop(cols[[0,1,2]], inplace=True, axis=1)

#fselection = {1: [0,1,2,6,7], 2: [0,1,2,3,4,5,6,7], 3: [0,1,2,4,5,7], 4: [4,5,6,7]}
fselection = {1: [0,1,4,6,7], 2: [0,1,2,5,7], 3: [0,6], 4: [0,2,3,4,5,6,7]}

#cor = ['Blue', 'Red', 'SWNIR_2']

cols = dataset1.iloc[:,1:].columns

X1 = dataset1[cols[fselection[1]]]
X2 = dataset2[cols[fselection[2]]]
X3 = dataset3[cols[fselection[3]]]
X4 = dataset4[cols[fselection[4]]]

Y1 = dataset1['Class']
Y2 = dataset2['Class']
Y3 = dataset3['Class']
Y4 = dataset4['Class']


#cor = ['Blue', 'Red', 'SWNIR_2']

# dataset1.drop(cor, inplace=True, axis=1)
# dataset2.drop(cor, inplace=True, axis=1)
# dataset3.drop(cor, inplace=True, axis=1)
# dataset4.drop(cor, inplace=True, axis=1)


def bmaPrediction2(sample1, sample2, sample3, sample4):

  with open('../TrainedModels/BMA/image1.BMAmodelFS.LogWeighted.pkl','r') as f:
    model1 = pickle.load(f)
  with open('../TrainedModels/BMA/image2.BMAmodelFS.LogWeighted.pkl','r') as f:
    model2 = pickle.load(f)
  with open('../TrainedModels/BMA/image3.BMAmodelFS.LogWeighted.pkl','r') as f:
    model3 = pickle.load(f)
  with open('../TrainedModels/BMA/image4.BMAmodelFS.LogWeighted.pkl','r') as f:
    model4 = pickle.load(f)

  w1 = model1.bmaweight
  w2 = model2.bmaweight
  w3 = model3.bmaweight
  w4 = model4.bmaweight

  s = float(w1+w2+w3+w4)
  w1 = w1/s
  w2 = w2/s
  w3 = w3/s
  w4 = w4/s

  # w1 = 1-w1
  # w2 = 1-w2
  # w3 = 1-w3
  # w4 = 1-w4
  
  p1 = pd.DataFrame(model1.predict(sample1, type='raw')*w1)
  p2 = pd.DataFrame(model2.predict(sample2, type='raw')*w2)
  p3 = pd.DataFrame(model3.predict(sample3, type='raw')*w3)
  p4 = pd.DataFrame(model4.predict(sample4, type='raw')*w4)
  
  currProbabilities = pd.concat([p1, p2, p3, p4], axis=0).reset_index(drop=True)

  currProbabilities = np.mean(currProbabilities, axis=0)
  c = model1.classes[np.argmax(currProbabilities)]
  return c


def bmaPrediction(sample, model1, model2, model3, model4):
  if (len(sample) != 4) :
    print("error in input")
    return None

  w1 = model1.bmaweight
  w2 = model2.bmaweight
  w3 = model3.bmaweight
  w4 = model4.bmaweight

  s = float(w1+w2+w3+w4)
  w1 = w1/s
  w2 = w2/s
  w3 = w3/s
  w4 = w4/s

  # w1 = 1-w1
  # w2 = 1-w2
  # w3 = 1-w3
  # w4 = 1-w4
  
  p1 = pd.DataFrame(model1.predict(sample.iloc[0,1:], type='raw')*w1)
  p2 = pd.DataFrame(model2.predict(sample.iloc[1,1:], type='raw')*w2)
  p3 = pd.DataFrame(model3.predict(sample.iloc[2,1:], type='raw')*w3)
  p4 = pd.DataFrame(model4.predict(sample.iloc[3,1:], type='raw')*w4)
  
  currProbabilities = pd.concat([p1, p2, p3, p4], axis=0).reset_index(drop=True)

  currProbabilities = np.mean(currProbabilities, axis=0)
  c = model1.classes[np.argmax(currProbabilities)]
  return c


# Partial attributes kept for modelling. Need to have appropriate mdoels saved in the folder.
def main2():
  #outputDataFrame = dataset1
  predictions = []
  for i in range(len(Y1)):
  #i = 0
    #sample = pd.concat([dataset1.iloc[i,], dataset2.iloc[i,], dataset3.iloc[i,], dataset4.iloc[i,]], axis=1).transpose().reset_index(drop=True)
    c = bmaPrediction2(X1.iloc[i,], X2.iloc[i,], X3.iloc[i,], X4.iloc[i,])
    predictions.append(c)
  accuracy = sum(np.array(predictions) == np.array(Y1)) / float(len(predictions))
  #tab = pd.crosstab(np.array(predictions), np.array(outputDataFrame['Class'])) 
  #print len(tab)
  print accuracy
  print class_accuracies(np.array(predictions), np.array(Y1))


# All attributes kept for modelling. Need to have appropriate mdoels saved in the folder.
def main1():
  with open('../TrainedModels/BMA/image1.BMAmodel.LogWeighted.pkl','r') as f:
    model1 = pickle.load(f)
  with open('../TrainedModels/BMA/image2.BMAmodel.LogWeighted.pkl','r') as f:
    model2 = pickle.load(f)
  with open('../TrainedModels/BMA/image3.BMAmodel.LogWeighted.pkl','r') as f:
    model3 = pickle.load(f)
  with open('../TrainedModels/BMA/image4.BMAmodel.LogWeighted.pkl','r') as f:
    model4 = pickle.load(f)

  outputDataFrame = dataset1
  predictions = []
  for i in range(len(Y1)):
    sample = pd.concat([dataset1.iloc[i,], dataset2.iloc[i,], dataset3.iloc[i,], dataset4.iloc[i,]], axis=1).transpose().reset_index(drop=True)
    c = bmaPrediction(sample, model1, model2, model3, model4)
    predictions.append(c)
  accuracy = sum(np.array(predictions) == np.array(outputDataFrame['Class'])) / float(len(predictions))
  #tab = pd.crosstab(np.array(predictions), np.array(outputDataFrame['Class'])) 
  #print len(tab)
  print accuracy
  print class_accuracies(np.array(predictions), np.array(Y1))

def main3():
  with open('../TrainedModels/BMA/pcaImage1.BMAmodel.LogWeighted.pkl','r') as f:
    model1 = pickle.load(f)
  with open('../TrainedModels/BMA/pcaImage2.BMAmodel.LogWeighted.pkl','r') as f:
    model2 = pickle.load(f)
  with open('../TrainedModels/BMA/pcaImage3.BMAmodel.LogWeighted.pkl','r') as f:
    model3 = pickle.load(f)
  with open('../TrainedModels/BMA/pcaImage4.BMAmodel.LogWeighted.pkl','r') as f:
    model4 = pickle.load(f)

  d1=pd.read_csv("../Data/Testing/pcaTestImage1.csv")
  d2=pd.read_csv("../Data/Testing/pcaTestImage2.csv")
  d3=pd.read_csv("../Data/Testing/pcaTestImage3.csv")
  d4=pd.read_csv("../Data/Testing/pcaTestImage4.csv")


  outputDataFrame = d1
  predictions = []
  for i in range(len(Y1)):
    sample = pd.concat([d1.iloc[i,], d2.iloc[i,], d3.iloc[i,], d4.iloc[i,]], axis=1).transpose().reset_index(drop=True)
    c = bmaPrediction(sample, model1, model2, model3, model4)
    predictions.append(c)
  accuracy = sum(np.array(predictions) == np.array(outputDataFrame['Class'])) / float(len(predictions))
  #tab = pd.crosstab(np.array(predictions), np.array(outputDataFrame['Class'])) 
  #print len(tab)
  print accuracy
  print class_accuracies(np.array(predictions), np.array(Y1))

if __name__ == '__main__':
  main2()
