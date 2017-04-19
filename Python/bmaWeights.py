import numpy as np
import pandas as pd
import pickle
from MLCFast import *

dataset1 = pd.read_csv('../Data/Training/ValidationDataImage1.csv')
dataset2 = pd.read_csv('../Data/Training/ValidationDataImage2.csv')
dataset3 = pd.read_csv('../Data/Training/ValidationDataImage3.csv')
dataset4 = pd.read_csv('../Data/Training/ValidationDataImage4.csv')

cols = dataset1.columns

dataset1.drop(cols[[0,1,2]], inplace=True, axis=1)
dataset2.drop(cols[[0,1,2]], inplace=True, axis=1)
dataset3.drop(cols[[0,1,2]], inplace=True, axis=1)
dataset4.drop(cols[[0,1,2]], inplace=True, axis=1)

cor = ['Blue', 'Red', 'SWNIR_2']

# dataset1.drop(cor, inplace=True, axis=1)
# dataset2.drop(cor, inplace=True, axis=1)
# dataset3.drop(cor, inplace=True, axis=1)
# dataset4.drop(cor, inplace=True, axis=1)


with open('../TrainedModels/MLC_Image1.pkl','r') as f:
  model1 = pickle.load(f)
with open('../TrainedModels/MLC_Image2.pkl','r') as f:
  model2 = pickle.load(f)
with open('../TrainedModels/MLC_Image3.pkl','r') as f:
  model3 = pickle.load(f)
with open('../TrainedModels/MLC_Image4.pkl','r') as f:
  model4 = pickle.load(f)


def calculateWeight (dataset, probs):
  weight = 1
  for i in range(len(dataset)):
    index = int(dataset.iloc[i,0])
    weight = weight*probs.iloc[i,(index-1)]
  return (weight)

def calculateLogWeight(dataset, probs):
  weight = 0
  for i in range(len(dataset)):
    index = int(dataset.iloc[i,0])
    weight = weight + np.log(1.0+probs.iloc[i,(index-1)])
    #print probs.iloc[i,(index-1)]
  return (weight)

def saveModel(model,dataset,filename):
  predProbs = model.predict(dataset1.iloc[:,1:],type = 'raw')
  w = calculateLogWeight(dataset, predProbs)
  print(w)
  model.bmaweight = w
  with open(filename,'wb') as f:
    pickle.dump(model, f)

saveModel(model4,dataset4,'../TrainedModels/image4.BMAmodel.LogWeighted.pkl')
