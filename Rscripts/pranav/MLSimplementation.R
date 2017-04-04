
calculateProb <- function(classData, dataPoint, classPriori){
  # create covariance matrix of data
  covariance = cov(classData)
  # compute means of all the columns
  mean = colMeans(classData)
  
  # calculate probability
  prob = (exp(-(as.matrix(dataPoint-mean))%*%solve(covariance)%*%(t(dataPoint-mean))))*classPriori/sqrt(det(covariance))
  print(prob)
  
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
    print(class1Prob)
    class2Prob = calculateProb(class2Data, tData[i,], class2Priori)
    print(class2Prob)
    class3Prob = calculateProb(class3Data, tData[i,], class3Priori)
    print(class3Prob)
    class4Prob = calculateProb(class4Data, tData[i,], class4Priori)
    print("###class 4###")
    print(class4Prob)
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
dataset = read.csv("../../TrainingData/ValidationData/ValidationData-2015-04-19.csv")

# create a test vector after shuffling the data
shuffleVec = sample(nrow(dataset),nrow(dataset))
testVector = sample(nrow(dataset),nrow(dataset)*0.2)
testData = dataset[testVector,]

# create traindata by eliminating the test data
trainData = dataset[-testVector,]

# delete first 4 columns and get only band values in bands variable
bandsWithClass = trainData[-c(1,2,3)]

# get predicted class list for test data
prediction = predict(bandsWithClass, testData[-c(1,2,3,4)])

# append this prediction to existing test data
testData[["Prediction"]] <- prediction

bands = bandsWithClass[-1]



# create a map for parameters of individual bands
#list = c()

# iterate over each band and store the respective parameters in the map
#for(i in names(bands))
#  params = c(mean(bands$i), sd(bands$i))
#  list$i = params
  