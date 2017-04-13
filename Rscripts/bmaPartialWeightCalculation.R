library(e1071)

dataset1=read.csv("../Data/Training/ValidationDataImage1.csv")
dataset2=read.csv("../Data/Training/ValidationDataImage2.csv")
dataset3=read.csv("../Data/Training/ValidationDataImage3.csv")
dataset4=read.csv("../Data/Training/ValidationDataImage4.csv")
dataset1 = dataset1[,-c(1,2,3)]
dataset2 = dataset2[,-c(1,2,3)]
dataset3 = dataset3[,-c(1,2,3)]
dataset4 = dataset4[,-c(1,2,3)]

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
  w = calculateLogWeight(datasetWithProbs)
  print(w)
  image4.BMAmodel.LogWeighted = list(model[[1]],model[[2]],w)
  save(image4.BMAmodel.LogWeighted,file=filename)
}

saveModel(image4.MLCmodel,dataset4,'image4.BMAmodel.LogWeighted.rda')
