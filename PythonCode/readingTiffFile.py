import threading
import gdal

dataset=gdal.Open("2015-04-19-AllBands-Clipped.TIF")
geotransform = dataset.GetGeoTransform()

band = dataset.GetRasterBand(1)

cols = dataset.RasterXSize
rows = dataset.RasterYSize
bands = dataset.RasterCount
driver = dataset.GetDriver().LongName
print cols,rows,bands,driver

band=[]
data=[]
for i in range(0,bands):
    currentBand=dataset.GetRasterBand(i+1);
    band.append(currentBand)
    data.append(currentBand.ReadAsArray(0,0,cols,rows))

class ReadFile (threading.Thread):
    def __init__(self, threadID, name,x,rangex):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.x = x
        self.rangex=rangex
        print

    def run(self):
        printBandValue(self.name, self.x,self.rangex)

def printBandValue(threadName, x,rangex):
    for xcod in range(x,x+rangex):
        for ycod in range(0, rows):
            for bandNumber in range(0,bands):
                print data[bandNumber][xcod,ycod],
            print "\n"


for i in range(0,21):
    ReadFile(1,"thread"+str(i),100*i,100).start()