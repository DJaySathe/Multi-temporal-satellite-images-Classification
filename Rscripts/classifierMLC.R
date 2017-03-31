library(e1071)
dataset = read.csv("ValidationData-2015-04-19.csv")
dataset = dataset[,-1]
shuffleVec = sample(nrow(dataset),nrow(dataset))
testVector <- sample(nrow(dataset),nrow(dataset)*0.2)

#[1]  21  97 143  77 112  12 124  10  95 149  52 153  57 162  41 103 116 155  29  70 152
#[22]   2  37  53  42 172  65  67 166  58   9  56 140  71  24  91

testData <- dataset[testVector,]
trainData <- dataset[-testVector,]
trainData = trainData[,-c(1,2)]
testData = testData[,-c(1,2)]
rownames(trainData) = NULL
rownames(testData) = NULL
image2.MLCmodel = naiveBayes(as.factor(trainData$Class)~., data = trainData)
table(predict(image1.MLCmodel,testData[,-1]),testData[,1])
err = sum(predict(image1.MLCmodel,testData[,-1])!=testData[,1])
acc=err/nrow(testData)
print(1-acc)
save(image1.MLCmodel,file = "image1.MLCmodel.rda")
#output: accuraccy 100%
#   1  2  3  4
# 1 10  0  0  0
# 2  0 13  0  0
# 3  0  0  6  0
# 4  0  0  0  7

accuracyTestData = read.csv("C:/Users/Sanket Shahane/Google Drive/MS/ALDA/Project/Multi-Temporal-Classification/TrainingData/AccuracyTestingData/AccuracyData-2015-12-31.csv")
accuracyTestData = accuracyTestData[,-c(1,2,3)]
head(accuracyTestData)
rownames(accuracyTestData) = NULL
table(predict(image2.MLCmodel,accuracyTestData[,-1]),accuracyTestData[,1])
err = sum(predict(image2.MLCmodel,testData[,-1])!=testData[,1])
acc=err/nrow(testData)
print(1-acc)
#  1  2  3  4
#1 12  0  0  0
#2  0 13  0  0
#3  0  0 11  1
#4  0  0  1  7


#trying logistic regression

model.logistic = glm(as.factor(trainData$Class)~.,data=trainData,family= binomial(link="logit"))

library(nnet)
model.logistic = multinom(trainData$Class~.,data=trainData)
prediction = predict(model.logistic,testData[,-1],"probs")





# Code for saving the classifier on each image after validation.

library(e1071)
dataset = read.csv("ValidationData-2015-04-19.csv") #open the image data
dataset = dataset[,-1]
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
  
  temp.model = naiveBayes(as.factor(trainData$Class)~.,data=trainData)
  temp.predictions = predict(temp.model,testData[,-1])
  tmp.err = sum(temp.predictions!=testData[,1])/nrow(testData)
  print(table(predict(temp.model,testData[,-1]),testData[,1]))
  print(tmp.err)
  crossvalidationError = crossvalidationError+tmp.err
}
crossvalidationError = crossvalidationError/k
model = naiveBayes(as.factor(dataset$Class)~.,data=dataset)

image1.MLCmodel = list(model,crossvalidationError)
save(image1.MLCmodel,file = "image1.MLCmodel.rda") #save the model and the error results rename the file according to the image

image2.MLCmodel = naiveBayes(as.factor(trainData$Class)~., data = trainData)
table(predict(image1.MLCmodel,testData[,-1]),testData[,1])
err = sum(predict(image1.MLCmodel,testData[,-1])!=testData[,1])
acc=err/nrow(testData)









