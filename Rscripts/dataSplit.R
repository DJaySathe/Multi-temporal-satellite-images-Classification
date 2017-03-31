dataset = read.csv("finaldata-2015-04-19.csv")
head(dataset)

accuracyTestData = sample(nrow(dataset),nrow(dataset)*0.2)

accuracyDataVector = sample(nrow(dataset),nrow(dataset)*0.2)
accuracyTestData = dataset[accuracyDataVector,]
df = dataset[-accuracyDataVector,]
write.csv(df,file = "ValidationData-2015-04-19.csv")
write.csv(accuracyTestData,file = "AccuracyData-2015-04-19.csv")

