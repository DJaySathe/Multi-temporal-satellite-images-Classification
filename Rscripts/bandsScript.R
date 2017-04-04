plotBandsPerClass = function(class1Data,title)
{
  class1.densityUltra_Blue = density(class1Data$Ultra_Blue)
  class1.densityBlue = density(class1Data$Blue)
  class1.densityGreen = density(class1Data$Green)
  class1.densityRed = density(class1Data$Red)
  class1.densityNIR = density(class1Data$NIR)
  class1.densitySWNIR_1 = density(class1Data$SWNIR_1)
  class1.densitySWNIR_2 = density(class1Data$SWNIR_2)
  class1.densityCirrus = density(class1Data$Cirrus)
  
  plot(class1.densityUltra_Blue,col="Dark Blue",xlim=c(4000,17000),main=title)
  lines(class1.densityBlue,col="Light Blue")
  lines(class1.densityGreen,col="Green")
  lines(class1.densityRed,col="Red")
  lines(class1.densityNIR,col="Orange")
  lines(class1.densitySWNIR_1,col="Purple")
  lines(class1.densitySWNIR_2,col="Pink")
  lines(class1.densityCirrus,col="black")
}

class1Data = originalData[originalData$Class==1,]
plotBandsPerClass(class1Data,"Class 1: Open Lands")
class1Data = originalData[originalData$Class==2,]
plotBandsPerClass(class1Data,"Class 2: Settlements")
class1Data = originalData[originalData$Class==3,]
plotBandsPerClass(class1Data,"Class 3: Vegetation")

#compare band one in all classes:

compareClasses <- function(originalData,band){
  temp = band
  band=band+3
  partialData = originalData[originalData$Class==1,]
  c1.densityBand = density(partialData[,band])
  partialData = originalData[originalData$Class==2,]
  c2.densityBand = density(partialData[,band])
  partialData = originalData[originalData$Class==3,]
  c3.densityBand = density(partialData[,band])
  partialData = originalData[originalData$Class==4,]
  c4.densityBand = density(partialData[,band])
  
  plot(c1.densityBand,col="Red",main=paste("All classes: Band",temp))
  lines(c2.densityBand,col="Yellow")
  lines(c3.densityBand,col="Green")
  lines(c4.densityBand,col="Blue")
}

#Change the csv location for the code to run on data from other images.
originalData = read.csv("finaldata-2015-12-31.csv")
#this line needed if the csv does not have proper band names
#colnames(originalData) = c("X","Y","Class","Ultra_Blue","Blue","Green","Red","NIR","SWNIR_1","SWNIR_2","Cirrus")
compareClasses(originalData,1)
compareClasses(originalData,2)
compareClasses(originalData,3)
compareClasses(originalData,4)
compareClasses(originalData,5)
compareClasses(originalData,6)
compareClasses(originalData,7)
compareClasses(originalData,8)