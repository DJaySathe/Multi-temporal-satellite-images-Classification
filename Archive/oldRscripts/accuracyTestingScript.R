setwd("C:/Users/Sanket Shahane/Google Drive/MS/ALDA/Project/Multi-Temporal-Classification/Rscripts")
load("image1.MLCmodel.rda")
load("image2.MLCmodel.rda")
load("image3.MLCmodel.rda")
load("image4.MLCmodel.rda")

setwd("C:/Users/Sanket Shahane/Google Drive/MS/ALDA/Project/Multi-Temporal-Classification/TrainingData/AccuracyTestingData")
dataset = read.csv("AccuracyData-2016-03-20.csv")
dataset = dataset[,-c(1,2,3)]
head(dataset)

current.model = image4.MLCmodel[[1]]

table(predict(current.model,dataset[,-1]),dataset[,1])
accuracy = sum(predict(current.model,dataset[,-1])==dataset[,1])
accuracy = accuracy/nrow(dataset)
print(accuracy)
print(paste("Expected prediction accuracy:",(1-image4.MLCmodel[[2]])))