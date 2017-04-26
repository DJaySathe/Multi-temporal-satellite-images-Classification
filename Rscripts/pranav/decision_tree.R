# Random Forest
library(randomForest)
dataset1 = read.csv("../../Data/Training/ValidationDataImage1.csv")
training_data1 = dataset[,-c(1,2,3)]
model.rf1 = randomForest(as.factor(training_data1$Class)~.,data=training_data1[,-4], ntree=9)
model.rf1$importance

dataset2 = read.csv("../../Data/Training/ValidationDataImage2.csv")
training_data2 = dataset[,-c(1,2,3)]
model.rf2 = randomForest(as.factor(training_data1$Class)~.,data=training_data1[,-4], ntree=9)
model.rf2$importance

dataset3 = read.csv("../../Data/Training/ValidationDataImage3.csv")
training_data3 = dataset[,-c(1,2,3)]
model.rf3 = randomForest(as.factor(training_data1$Class)~.,data=training_data1[,-4], ntree=9)
model.rf3$importance

dataset4 = read.csv("../../Data/Training/ValidationDataImage4.csv")
training_data4 = dataset[,-c(1,2,3)]
model.rf4 = randomForest(as.factor(training_data1$Class)~.,data=training_data1[,-4], ntree=9)
model.rf4$importance

#prediction.rf = predict(model.rf1,test_data[,-4])
#sum(prediction.rf==test_data[,4])/nrow(test_data)