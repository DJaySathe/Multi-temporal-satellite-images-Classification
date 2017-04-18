import pandas as pd
import numpy as np
import pickle


class MLC:
  def __init__(self):
    self.priorProb = []
    self.classes = None
    self.means = []
    self.covs = []
    return

  def compute_apriori (self, df):
    counts = df.value_counts(sort=False)
    prior = counts/len(df)
    return np.array(prior)

  def compute_means(self, df):
    return np.matrix(np.mean(df, axis=0))

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
    likelihood = float((X - self.means[c])*np.linalg.inv(cv)*np.transpose(X - self.means[c]))
    likelihood = np.exp(-likelihood)/np.sqrt(np.linalg.det(cv))
    return prior*likelihood


  def predict(self, testX):
    N = len(testX)
    predictions = []
    for i in range(N):
      probs = []
      for j in range(len(self.classes)):
        t = np.matrix(testX.loc[i])
        probs.append(self.calcProb(t, j))
      predictions.append(self.classes[np.argmax(probs)])
    return predictions

  def score(self, predictions, actual):
    accuracy = sum(predictions == actual)/float(len(actual))
    return accuracy




if __name__ == "__main__":
  data = pd.read_csv('../Data/Training/ValidationDataImage4.csv')
  cols = data.columns
  data.drop(cols[[0,1,2]], inplace=True, axis=1)
  X = data.iloc[:,1:]
  y = data['Class']

  accuracyTestData = pd.read_csv("../Data/Testing/AccuracyDataImage4.csv")
  cols = accuracyTestData.columns
  accuracyTestData.drop(cols[[0,1,2]], inplace=True, axis=1)
  Xtest = accuracyTestData.iloc[:,1:]
  ytest = accuracyTestData['Class']

  model = MLC()
  model.fit(X, y)

  preds = model.predict(Xtest)
  accuracy = model.score(preds, ytest)
  print accuracy

  with open('../TrainedModels/MLC_Image4.pkl','wb') as f:
    pickle.dump(model,f)

