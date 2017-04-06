set.seed(100)

#accuracyDataVector = sample(nrow(dataset),nrow(dataset)*0.2)
temp = read.csv("AccuracyData-2015-04-19.csv")
accuracyDataVector = temp$X.1

setwd("C:/Users/Sanket Shahane/Google Drive/MS/ALDA/ImageClassification/Multi-temporal-Classification-of-satellite-images/TrainingData/Corrected Data")
dataset = read.csv("2015-04-19.csv")
head(dataset)
accuracyTestData = dataset[accuracyDataVector,]
df = dataset[-accuracyDataVector,]
write.csv(df,file = "ValidationData-2015-04-19.csv")
write.csv(accuracyTestData,file = "AccuracyData-2015-04-19.csv")

