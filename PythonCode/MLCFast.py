import pandas as pd
import numpy as np
import pickle


class MLCFast:
  def __init__(self):
    self.priorProb = []
    self.classes = None
    self.means = []
    self.covs = []
    self.predProbs = []
    self.bmaweight = 0.0
    return

  def compute_apriori (self, df):
    counts = df.value_counts(sort=False)
    prior = counts/len(df)
    return np.array(prior)

  def compute_means(self, df):
    return np.mean(df, axis=0)

  def compute_covs (self, df):
    return np.matrix(np.cov(np.transpose(df)))

  def fit(self, X, y):
    self.classes = np.unique(y)

    for c in self.classes:
      classData = X[y == c]
      self.priorProb.append(len(classData))
      mu = self.compute_means(classData)
      self.means.append(mu)
      sigma = self.compute_covs(classData)
      self.covs.append(sigma)
    self.priorProb = list(np.array(self.priorProb)/float(len(X)))

  def calcProb(self, X, c):
    prior = self.priorProb[c]
    cv = self.covs[c]
    T1 = np.matrix((X - self.means[c]))
    T2 = np.linalg.inv(cv)
    T3 = np.transpose(T1)
    likelihood = (T1*T2)
    likelihood = likelihood*T3
    likelihood = np.diag(likelihood)

    likelihood = np.exp(-likelihood)/np.sqrt(np.linalg.det(cv))
    return prior*likelihood

  def predict(self, testX, type='default'):
    N = len(testX)
    x = np.empty([len(testX),len(self.classes)])
    probList = pd.DataFrame(x)
    
    for j in range(len(self.classes)):
      prob = self.calcProb(testX, j)
      prob = np.array(prob).reshape(len(testX),1)
      prob = pd.DataFrame(prob)
      probList[j] = prob
    
    self.predProbs = probList
    if (type == 'default'):
      predIndex = np.argmax(np.matrix(probList), axis=1)
      predictions = list(pd.DataFrame(np.array(self.classes)[predIndex]).iloc[:,0])
      return predictions
    elif (type == 'raw'):
      return probList
    else:
      print ('Invalid type. Can be either default or raw')
    return None

  def score(self, predictions, actual):
    accuracy = sum(predictions == actual)/float(len(actual))
    return accuracy


if __name__ == "__main__":
  imageNumber = 4
  data = pd.read_csv('../Data/Training/ValidationDataImage'+str(imageNumber)+'.csv')
  cols = data.columns
  data.drop(cols[[0,1,2]], inplace=True, axis=1)
  cor = ['Blue', 'Red', 'SWNIR_1']
  #data.drop(cor, inplace=True, axis=1)
  X = data.iloc[:,1:]
  y = data['Class']

  accuracyTestData = pd.read_csv('../Data/Testing/AccuracyDataImage'+str(imageNumber)+'.csv')
  cols = accuracyTestData.columns
  accuracyTestData.drop(cols[[0,1,2]], inplace=True, axis=1)
  #accuracyTestData.drop(cor, inplace=True, axis=1)
  Xtest = accuracyTestData.iloc[:,1:]
  ytest = accuracyTestData['Class']

  model = MLCFast()
  model.fit(X, y)

  preds = model.predict(Xtest)
  accuracy = model.score(preds, ytest)
  print accuracy

  with open('../TrainedModels/MLC_Image'+str(imageNumber)+'.pkl','wb') as f:
    pickle.dump(model,f)

