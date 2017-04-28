
calculateProb <- function(classData, dataPoint, classPriori){
  # create covariance matrix of data
  covariance = cov(classData)
  # compute means of all the columns
  mean = colMeans(classData)
  # calculate probability
  prob = (exp(-(as.matrix(dataPoint-mean))%*%solve(covariance)%*%(t(dataPoint-mean))))*classPriori/sqrt(det(covariance))
  return(prob)
}

predict <- function(bandsWithClass, tData){
  prediction = c()
  # seperate out data based on class assignment
  class1Data = bandsWithClass[bandsWithClass$Class == 1,][,-1]
  class1Priori = nrow(class1Data)/nrow(bandsWithClass)
  class2Data = bandsWithClass[bandsWithClass$Class == 2,][,-1]
  class2Priori = nrow(class2Data)/nrow(bandsWithClass)
  class3Data = bandsWithClass[bandsWithClass$Class == 3,][,-1]
  class3Priori = nrow(class3Data)/nrow(bandsWithClass)
  class4Data = bandsWithClass[bandsWithClass$Class == 4,][,-1]
  class4Priori = nrow(class4Data)/nrow(bandsWithClass)
  
  for (i in c(1:nrow(tData))){
  
    # campute all class probabilities
    class1Prob = calculateProb(class1Data, tData[i,], class1Priori)
    #print(class1Prob)
    class2Prob = calculateProb(class2Data, tData[i,], class2Priori)
    #print(class2Prob)
    class3Prob = calculateProb(class3Data, tData[i,], class3Priori)
    #print(class3Prob)
    class4Prob = calculateProb(class4Data, tData[i,], class4Priori)
    #print("###class 4###")
    #print(class4Prob)
    finalClass = 1;
    finalProb = class1Prob
    if(class2Prob > finalProb){
      finalClass = 2
      finalProb = class2Prob
    }
    if(class3Prob > finalProb){
      finalClass = 3
      finalProb = class3Prob
    }
    if(class4Prob > finalProb){
      finalClass = 4
      finalProb = class4Prob
    }
    prediction = c(prediction, finalClass)
  }
  return(prediction)
}


library(MASS)
# set work directory as the location of the script

dataset = read.csv("../Data/Training/ValidationDataImage3.csv")
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
  
  prediction = predict(trainData, testData[,-1])
  temp.err = (sum(prediction != testData$Class)/nrow(testData))
  #temp.model = naiveBayes(as.factor(trainData$Class)~.,data=trainData)
  #temp.predictions = predict(temp.model,testData[,-1])
  #tmp.err = sum(temp.predictions!=testData[,1])/nrow(testData)
  #print(table(predict(temp.model,testData[,-1]),testData[,1]))
  print(temp.err)
  crossvalidationError = crossvalidationError+temp.err
}
print(crossvalidationError/10)


#Testing Dataset
testData = read.csv('../Data/Testing/AccuracyDataImage3.csv')
testData = testData[,-c(1,2,3)]

err = sum(predict(dataset,testData[,-1])!=testData[,1])
acc=err/nrow(testData)