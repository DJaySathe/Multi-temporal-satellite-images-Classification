library(e1071)

setwd("C:/Users/Sanket Shahane/Google Drive/MS/ALDA/Project/Multi-Temporal-Classification/TrainingData/ValidationData")
dataset1=read.csv("ValidationData-2015-04-19.csv")
dataset2=read.csv("ValidationData-2015-12-31.csv")
dataset3=read.csv("ValidationData-2016-01-16.csv")
dataset4=read.csv("ValidationData-2016-03-20.csv")
dataset1 = dataset1[,-c(1,2,3)]
dataset2 = dataset2[,-c(1,2,3)]
dataset3 = dataset3[,-c(1,2,3)]
dataset4 = dataset4[,-c(1,2,3)]

setwd("C:/Users/Sanket Shahane/Google Drive/MS/ALDA/Project/Multi-Temporal-Classification/Rscripts")
#load("image1.BMAmodel.rda")
#load("image2.BMAmodel.rda")
#load("image3.BMAmodel.rda")
#load("image4.BMAmodel.rda")

load("image1.BMAmodel.LogWeighted.rda")
load("image2.BMAmodel.LogWeighted.rda")
load("image3.BMAmodel.LogWeighted.rda")
load("image4.BMAmodel.LogWeighted.rda")

bmaPrediction <- function(sample,i){
  if(nrow(sample)!=4){
    print("error in input")
    return(NULL)
  }
  #p1 = data.frame(predict(image1.BMAmodel[[1]],sample[1,-1],type="raw"))*image1.BMAmodel[[3]]
  #p2 = data.frame(predict(image2.BMAmodel[[1]],sample[2,-1],type="raw"))*image2.BMAmodel[[3]]
  #p3 = data.frame(predict(image3.BMAmodel[[1]],sample[3,-1],type="raw"))*image3.BMAmodel[[3]]
  #p4 = data.frame(predict(image4.BMAmodel[[1]],sample[4,-1],type="raw"))*image4.BMAmodel[[3]]
  
  p1 = data.frame(predict(image1.BMAmodel.LogWeighted[[1]],sample[1,-1],type="raw"))*image1.BMAmodel.LogWeighted[[3]]
  p2 = data.frame(predict(image2.BMAmodel.LogWeighted[[1]],sample[2,-1],type="raw"))*image2.BMAmodel.LogWeighted[[3]]
  p3 = data.frame(predict(image3.BMAmodel.LogWeighted[[1]],sample[3,-1],type="raw"))*image3.BMAmodel.LogWeighted[[3]]
  p4 = data.frame(predict(image4.BMAmodel.LogWeighted[[1]],sample[4,-1],type="raw"))*image4.BMAmodel.LogWeighted[[3]]
  
  currProbabilities = rbind(p1,p2,p3,p4)
  currProbabilities = apply(currProbabilities,2,mean)
  #print(currProbabilities)
  class = as.integer(which.min(currProbabilities))
  #print(class)
  # write this class to a data frame
  return(class)
}

outputDataFrame = dataset1
outputDataFrame = cbind(outputDataFrame,data.frame(NaN))
for (i in seq(1,nrow(outputDataFrame),1)) {
  sample = rbind(dataset1[i,],dataset2[i,],dataset3[i,],dataset4[i,])
  class = bmaPrediction(sample,i)
  outputDataFrame[i,10] = class
}
sum(outputDataFrame[,1]==outputDataFrame[,10])/nrow(outputDataFrame)

#average accuracy
averageAccuracy = (1 - image1.BMAmodel.LogWeighted[[2]] + 1 - image2.BMAmodel.LogWeighted[[2]] + 1 - image3.BMAmodel.LogWeighted[[2]] + 1- image4.BMAmodel.LogWeighted[[2]])/4
#0.8958333