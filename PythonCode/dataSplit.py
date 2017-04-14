import pandas as pd
import random
import numpy as np

random.seed(100)

dataset = pd.read_csv('../Data/OriginalDataImage1.csv')
dataset.head()

accuracyDataVector = np.random.rand(len(dataset)) < 0.2

accuracyTestData = dataset.loc[accuracyDataVector,]
df = dataset.loc[~accuracyDataVector,]
df.to_csv('../Data/Training/ValidationDataImage1.csv')
df.to_csv('../Data/Testing/AccuracyDataImage1.csv')

temp = pd.read_csv("../Data/Testing/AccuracyDataImage1.csv")
accuracyDataVector = temp['X.1']