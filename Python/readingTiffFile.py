import threading
from multiprocessing import Process

import gdal
import pandas as pd
import numpy as np
import time
from MLC import MLC
from matplotlib import pyplot as plt

colorCom = [[165,42,42], [139, 141, 122], [0, 128, 0], [32, 178, 170]]


class ReadFile (threading.Thread):
    def __init__(self, threadID, name,x,rangex,output,outputt,imageRows,imageCols,datasetData,model,imageBands):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.x = x
        self.rangex=rangex
        self.output = output
        self.outputt = outputt
        self.imageRows=imageRows
        self.imageCols=imageCols
        self.datasetData=datasetData
        self.model=model
        self.imageBands=imageBands

    def run(self):
        printBandValue(self.name, self.x,self.rangex,
                       self.output,self.outputt,self.imageRows,
                       self.imageCols,self.datasetData,self.model,self.imageBands)

def printBandValue(threadName, x,rangex,output,outputt,imageRows,imageCols,datasetData,model,imageBands):
    for xcod in range(x,x+rangex):
        for ycod in range(0, imageRows):
            l=[]
            for bandNumber in range(0, imageBands):
                l.append(datasetData[bandNumber][ ycod,xcod])
            l=np.asarray(l)

            if len(np.where(l==0)[0])==8:
                continue
            preds = model.predictSingle(l)
            output[ycod,xcod]=colorCom[preds-1]
            outputt[ycod,xcod]=colorCom[0] if preds == 1 or preds ==3 else colorCom[preds-1]

millis = int(round(time.time() * 1000))


def classIdentification(imagelocation,datalocation):
    imageDataset = gdal.Open(imagelocation)
    geotransform = imageDataset.GetGeoTransform()

    imageBand = imageDataset.GetRasterBand(1)

    imageCols = imageDataset.RasterXSize
    imageRows = imageDataset.RasterYSize
    imageBands = imageDataset.RasterCount
    imageDriver = imageDataset.GetDriver().LongName
    print imageCols, imageRows, imageBands, imageDriver

    data = pd.read_csv(datalocation)
    cols = data.columns
    data.drop(cols[[0, 1, 2]], inplace=True, axis=1)
    X = data.iloc[:, 1:]
    y = data['Class']
    model = MLC()
    model.fit(X, y)

    imageBand = []
    datasetData = []
    for i in range(0, imageBands):
        currentBand = imageDataset.GetRasterBand(i + 1);
        imageBand.append(currentBand)
        datasetData.append(currentBand.ReadAsArray(0, 0, imageCols, imageRows))

    output = np.zeros(shape=(imageRows, imageCols, 3))
    outputt = np.zeros(shape=(imageRows, imageCols, 3))

    for i in range(1,41):
        t=ReadFile(1,"thread"+str(i),50*i,50,output,outputt,imageRows,imageCols,datasetData,model,imageBands)
        t.start()
        t.join()

    fig=plt.figure(0)
    fig.canvas.set_window_title(imagelocation)
    plt.imshow(output, interpolation='nearest')
    plt.show()
    plt.imshow(outputt, interpolation='nearest')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    p1 = Process(target=classIdentification, args=(
        "../satellite images/2016-03-20-AllBands-Clipped.tif","../Data/Training/ValidationDataImage4.csv"))
    p1.start()
    p2 = Process(target=classIdentification, args=(
    "../satellite images/2016-01-16-AllBands-Clipped.tif", "../Data/Training/ValidationDataImage3.csv"))
    p2.start()
    p3 = Process(target=classIdentification, args=(
    "../satellite images/2015-12-31-AllBands-Clipped.tif", "../Data/Training/ValidationDataImage2.csv"))
    p3.start()
    p4 = Process(target=classIdentification, args=(
        "../satellite images/2015-04-19-AllBands-Clipped.tif", "../Data/Training/ValidationDataImage1.csv"))
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()