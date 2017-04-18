import threading
import gdal
import pandas as pd
import numpy as np
import pickle
from MLC import MLC
from matplotlib import pyplot as plt


imageDataset=gdal.Open("../satellite images/2016-03-20-AllBands-Clipped.tif")
geotransform = imageDataset.GetGeoTransform()

imageBand = imageDataset.GetRasterBand(1)

imageCols = imageDataset.RasterXSize
imageRows = imageDataset.RasterYSize
imageBands = imageDataset.RasterCount
imageDriver = imageDataset.GetDriver().LongName
print imageCols,imageRows,imageBands,imageDriver

output=np.zeros(shape=(imageRows,imageCols))
outputt=np.zeros(shape=(imageRows,imageCols))

data = pd.read_csv('../Data/Training/ValidationDataImage4.csv')
cols = data.columns
data.drop(cols[[0,1,2]], inplace=True, axis=1)
cor = ['Blue', 'Red', 'SWNIR_1']
#data.drop(cor, inplace=True, axis=1)
X = data.iloc[:,1:]
y = data['Class']
model = MLC()
model.fit(X, y)


imageBand=[]
datasetData=[]
for i in range(0, imageBands):
    currentBand=imageDataset.GetRasterBand(i + 1);
    imageBand.append(currentBand)
    datasetData.append(currentBand.ReadAsArray(0, 0, imageCols, imageRows))

class ReadFile (threading.Thread):
    def __init__(self, threadID, name,x,rangex):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.x = x
        self.rangex=rangex

    def run(self):
        printBandValue(self.name, self.x,self.rangex)

def printBandValue(threadName, x,rangex):
    for xcod in range(x,x+rangex):
        for ycod in range(0, imageRows):
            l=[]
            for bandNumber in range(0, imageBands):
                l.append(datasetData[bandNumber][ ycod,xcod])
            l=np.asarray(l)

            if len(np.where(l==0)[0])==8:
                continue
            preds = model.predictSingle(l)
            output[ycod,xcod]=preds*40
            outputt[ycod,xcod]=200 if preds == 1 or preds ==3 else preds *40


for i in range(1,20):
    t=ReadFile(1,"thread"+str(i),100*i,100)
    t.start()
    t.join()
plt.imshow(output, interpolation='nearest')
plt.show()
plt.imshow(outputt, interpolation='nearest')
plt.legend()
plt.show()
