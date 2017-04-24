import threading
from multiprocessing import Process

import gdal
import pandas as pd
import numpy as np
import time
from MLC import MLC
from matplotlib import pyplot as plt
import time


colorCom = [[165,42,42], [139, 141, 122], [0, 128, 0], [32, 178, 170]]


class ReadFile (threading.Thread):
    def __init__(self, threadID, name,output,outputt,imageRows,imageCols,datasetData,model,imageBands):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
#        self.x = x
#        self.rangex=rangex
        self.output = output
        self.outputt = outputt
        self.imageRows=imageRows
        self.imageCols=imageCols
        self.datasetData=datasetData
        self.model=model
        self.imageBands=imageBands

    def run(self):
        computeClass(self.name,
                     self.output, self.outputt, self.imageRows,
                     self.imageCols, self.datasetData, self.model, self.imageBands)

def computeClass(threadName, output, outputt, imageRows, imageCols, datasetData, model, imageBands):
    for ycod in range(0, len(datasetData[0])):
        if ycod%100==0:
            print ycod
        for xcod in range(0, len(datasetData[0][0])):
            l=[]
            for bandNumber in range(0, imageBands):
                l.append(datasetData[bandNumber][ ycod,xcod])
            l=np.asarray(l)

            if len(np.where(l==0)[0])==8:
                continue
            preds = model.predictSingle(l)
#            output[ycod,xcod]=colorCom[preds-1]
#            outputt[ycod,xcod]=colorCom[0] if preds == 1 or preds ==3 else colorCom[preds-1]

millis = int(round(time.time() * 1000))


def classIdentification(threadSize,imagelocation,datalocation):
    startTime = time.time()
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
    maxxRange=np.ceil(float(imageCols)/float(threadSize))
    maxxRange=int(maxxRange)
    d=[]
    for j in range(0, threadSize):
        for i in range(0, imageBands):
            currentBand = imageDataset.GetRasterBand(i + 1);
            imageBand.append(currentBand)
            maxxRange1 = maxxRange if maxxRange * j + maxxRange < imageCols else imageCols - maxxRange * j
            datasetData.append(currentBand.ReadAsArray(maxxRange*j, 0, maxxRange1, imageRows))
        d.append(datasetData)

    output = np.zeros(shape=(imageRows, imageCols, 3))
    outputt = np.zeros(shape=(imageRows, imageCols, 3))



    for i in range(0,threadSize):
        t=ReadFile(1,"thread"+str(i),output,outputt,imageRows,imageCols,d[i],model,imageBands)
        t.start()
        t.join()

    print "with threads ",str(pow(2,i))," total time required is",time.time()-startTime
#    fig=plt.figure(0)
#    fig.canvas.set_window_title(imagelocation)
#    plt.imshow(output, interpolation='nearest')
#    plt.show()
#    plt.imshow(outputt, interpolation='nearest')
#    plt.legend()
#    plt.show()

if __name__ == '__main__':
    for i in range(0,8):
        p1 = Process(target=classIdentification, args=(pow(2,i),"../satellite images/2016-03-20-AllBands-Clipped.tif","../Data/Training/ValidationDataImage4.csv"))
        p1.start()
        p1.join()

    '''
    p2 = Process(target=classIdentification, args=(
    "../satellite images/2016-01-16-AllBands-Clipped.tif", "../Data/Training/ValidationDataImage3.csv"))
    p2.start()
    p3 = Process(target=classIdentification, args=(
    "../satellite images/2015-12-31-AllBands-Clipped.tif", "../Data/Training/ValidationDataImage2.csv"))
    p3.start()
    p4 = Process(target=classIdentification, args=(
        "../satellite images/2015-04-19-AllBands-Clipped.tif", "../Data/Training/ValidationDataImage1.csv"))
    p4.start()

    p2.join()
    p3.join()
    p4.join()
    '''