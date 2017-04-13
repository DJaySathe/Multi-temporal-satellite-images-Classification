# Code for saving the classifier on each image after validation.

library(e1071)
setwd("C:/Users/Sanket Shahane/Google Drive/MS/ALDA/Project/Multi-Temporal-Classification/TrainingData/ValidationData")
dataset = read.csv("ValidationData-2016-03-20.csv") #open the image data
dataset = dataset[,-c(1,2,3)]
# shuffle dataset for cross-validation
shuffleVec = sample(nrow(dataset),nrow(dataset))
dataset = dataset[shuffleVec,]

crossvalidationError = 0
k=10
for(i in seq(0,k-1,1)){
  testVector = seq(1,nrow(dataset)%/%k)
  testVector = testVector + nrow(dataset)%/%k*i
  testData = dataset[testVector,]
  trainData = dataset[-testVector,]
#  readGdal
  temp.model = naiveBayes(as.factor(trainData$Class)~.,data=trainData)
  temp.predictions = predict(temp.model,testData[,-1])
  tmp.err = sum(temp.predictions!=testData[,1])/nrow(testData)
  print(table(predict(temp.model,testData[,-1]),testData[,1]))
  print(tmp.err)
  crossvalidationError = crossvalidationError+tmp.err
}
crossvalidationError = crossvalidationError/k
model = naiveBayes(as.factor(dataset$Class)~.,data=dataset)

#save the model and the error results rename the file according to the image
image4.MLCmodel = list(model,crossvalidationError)
save(image4.MLCmodel,file = "image4.MLCmodel.rda")