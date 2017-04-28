import threading
from multiprocessing import Process

import gdal
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import pickle



colorCom = [[165,42,42], [139, 141, 122], [0, 128, 0], [32, 178, 170]]
with open('../TrainedModels/BMA/image1.BMAmodel.LogWeighted.pkl', 'r') as f:
    model1 = pickle.load(f)
with open('../TrainedModels/BMA/image2.BMAmodel.LogWeighted.pkl', 'r') as f:
    model2 = pickle.load(f)
with open('../TrainedModels/BMA/image3.BMAmodel.LogWeighted.pkl', 'r') as f:
    model3 = pickle.load(f)
with open('../TrainedModels/BMA/image4.BMAmodel.LogWeighted.pkl', 'r') as f:
    model4 = pickle.load(f)

#fselection = {1: [0,1,4,6,7], 2: [0,1,2,5,7], 3: [0,6], 4: [0,2,3,4,5,6,7]}

def bmaPrediction2(sample1, sample2, sample3, sample4):

    w1 = model1.bmaweight
    w2 = model2.bmaweight
    w3 = model3.bmaweight
    w4 = model4.bmaweight

    s = float(w1 + w2 + w3 + w4)
    w1 = w1 / s
    w2 = w2 / s
    w3 = w3 / s
    w4 = w4 / s

    # w1 = 1-w1
    # w2 = 1-w2
    # w3 = 1-w3
    # w4 = 1-w4

    p1 = pd.DataFrame(model1.predict(sample1, type='raw') * w1)
    p2 = pd.DataFrame(model2.predict(sample2, type='raw') * w2)
    p3 = pd.DataFrame(model3.predict(sample3, type='raw') * w3)
    p4 = pd.DataFrame(model4.predict(sample4, type='raw') * w4)

    currProbabilities = pd.concat([p1, p2, p3, p4], axis=0).reset_index(drop=True)

    currProbabilities = np.mean(currProbabilities, axis=0)
    c = model1.classes[np.argmax(currProbabilities)]
    return c


def loadImage(imagelocation,threadSize):
    imageDataset = gdal.Open(imagelocation)
    imageCols = imageDataset.RasterXSize
    imageRows = imageDataset.RasterYSize
    imageBands = imageDataset.RasterCount
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

    return d



class ReadFile (threading.Thread):
    def __init__(self, threadID, name,output,outputt,datasetData,size):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.output = output
        self.outputt = outputt
        self.datasetData=datasetData
        self.size=size

    def run(self):
        computeClass(self.threadID,
                     self.output, self.outputt, self.datasetData,self.size)

def computeClass(threadID, output, outputt, datasetData,size):
    for ycod in range(0, len(datasetData[0][0])):
        if ycod%100==0:
            print ycod
        for xcod in range(0, len(datasetData[0][0][0])):
            if xcod % 100 == 0:
                print ycod,xcod

            l4=[]
            for iimage in range(0,4):
                l = []
                for bandNumber in range(0, 8):
                    l.append(datasetData[iimage][bandNumber][ycod,xcod])
                l4.append(l)
            l4=np.asarray(l4)
            if len(np.where(l4[0]==0)[0])==8:
                continue
            output[ycod,size*threadID+xcod]=colorCom[bmaPrediction2(l4[0],l4[1],l4[2],l4[3])]
#            outputt[ycod,xcod]=colorCom[0] if preds == 1 or preds ==3 else colorCom[preds-1]

millis = int(round(time.time() * 1000))


def classIdentification(threadSize,imagelocation1,imagelocation2,imagelocation3,imagelocation4):

    d1=loadImage(imagelocation1,threadSize)
    d2=loadImage(imagelocation2,threadSize)
    d3=loadImage(imagelocation3,threadSize)
    d4=loadImage(imagelocation4,threadSize)


    output = np.zeros(shape=(len(d1[1][0]), len(d1[0][0][0])*threadSize, 3))
    outputt = np.zeros(shape=(len(d1[1][0]), len(d1[0][0][0])*threadSize, 3))

    startTime = time.time()
    threads=[]


    for i in range(0,threadSize):
        d = []
        d.append(d1[i])
        d.append(d2[i])
        d.append(d3[i])
        d.append(d4[i])
        t=ReadFile(1,"thread"+str(i),output,outputt,d,threadSize)
        t.start()
        threads.append(t)

    for i in range(0,threadSize):
        threads[i].join()

    print "with threads ",str(pow(2,i))," total time required is",time.time()-startTime

    fig=plt.figure(0)
    fig.canvas.set_window_title(imagelocation1)
    plt.imshow(output, interpolation='nearest')
    plt.show()
    plt.imshow(outputt, interpolation='nearest')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    p1 = Process(target=classIdentification, args=(pow(2,6),"../satellite images/2016-03-20-AllBands-Clipped.tif","../satellite images/2016-01-16-AllBands-Clipped.tif"
                                                   ,"../satellite images/2015-12-31-AllBands-Clipped.tif","../satellite images/2015-04-19-AllBands-Clipped.tif"))
    p1.start()
    p1.join()