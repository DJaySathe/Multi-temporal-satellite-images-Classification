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
load("image1.MLCmodel.rda")
load("image2.MLCmodel.rda")
load("image3.MLCmodel.rda")
load("image4.MLCmodel.rda")

# function to calculate the weight of the model
calculateWeight <- function(dataset){
  weight = 1
  for(i in seq(1,nrow(dataset),1)){
    index = as.integer(dataset[i,1])
    weight = weight*dataset[i,index+9]
  }
  return(weight)
}

calculateLogWeight <- function(dataset){
  weight = 0
  for(i in seq(1,nrow(dataset),1)){
    index = as.integer(dataset[i,1])
    weight = weight + log(dataset[i,index+9])
  }
  return(weight)
}

calculateRelativeWeight <- function(dataset){
  weight = array(0,4)
  for (i in seq(1,nrow(dataset),1)) {
    index = as.integer(dataset[i,1])
    weight[index] = weight[index] + log(dataset[i,index+9])
  }
  print(weight)
  return(weight)
}

saveModel <- function(model,dataset,filename){
  image.probs = predict(model[[1]],dataset[,-1],type = "raw")
  head(image.probs)
  datasetWithProbs = cbind(dataset,image.probs)
  head(datasetWithProbs)
  w = calculateRelativeWeight(datasetWithProbs)
  print(w)
  image4.BMAmodel.RelativeLogWeighted = list(model[[1]],model[[2]],w)
  save(image4.BMAmodel.RelativeLogWeighted,file=filename)
}

saveModel(image4.MLCmodel,dataset4,'image4.BMAmodel.RelativeLogWeighted.rda')