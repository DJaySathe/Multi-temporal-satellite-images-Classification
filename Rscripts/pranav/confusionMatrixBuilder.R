library(e1071)
trainData = read.csv("../TrainingData/ValidationData/ValidationData-2016-03-20.csv")
testData = read.csv("../TrainingData/AccuracyTestingData/AccuracyData-2016-03-20.csv")
rownames(trainData) = NULL
rownames(testData) = NULL
image1.MLCmodel = naiveBayes(as.factor(trainData$Class)~., data = trainData)

#initialize confusion matrix
confusionMatrix = matrix(c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),ncol=4,byrow=TRUE)
colnames(confusionMatrix) <- c("1","2","3","4")
rownames(confusionMatrix) <- c("1","2","3","4")

for(i in c(1:nrow(testData))){
  predictedClass = predict(image1.MLCmodel, testData[i,-1])
  confusionMatrix[testData[i,]$Class, predictedClass] =  confusionMatrix[testData[i,]$Class, predictedClass] + 1
}

print(confusionMatrix)
