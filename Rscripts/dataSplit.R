dataset = read.csv("finaldata-2015-04-19.csv")
head(dataset)

accuracyTestData = sample(nrow(dataset),nrow(dataset)*0.2)
#> accuracyTestData
#[1] 228  65  97  93 104  51 144  28  40  63 148 117 109  23 188 198  70 126 152 132 224
#[22] 177  85 172  35  56 197  99 168  95 142 130  58 102  30 175 113  37 202  48  27 157
#[43]  76 119 112
accuracyDataVector = sample(nrow(dataset),nrow(dataset)*0.2)
accuracyTestData = dataset[accuracyDataVector,]
df = dataset[-accuracyDataVector,]
write.csv(df,file = "ValidationData-2015-04-19.csv")
write.csv(accuracyTestData,file = "AccuracyData-2015-04-19.csv")

