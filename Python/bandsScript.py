import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

def plotBandsPerClass(class1Data,title):
    class1DensityUltra_Blue = scipy.stats.gaussian_kde(class1Data['Ultra_Blue'],bw_method=None)
    class1DensityBlue = scipy.stats.gaussian_kde(class1Data['Blue'],bw_method=None)
    class1DensityGreen = scipy.stats.gaussian_kde(class1Data['Green'],bw_method=None)
    class1DensityRed = scipy.stats.gaussian_kde(class1Data['Red'],bw_method=None)
    class1DensityNIR = scipy.stats.gaussian_kde(class1Data['NIR'],bw_method=None)
    class1DensitySWNIR_1 = scipy.stats.gaussian_kde(class1Data['SWNIR_1'],bw_method=None)
    class1DensitySWNIR_2 = scipy.stats.gaussian_kde(class1Data['SWNIR_2'],bw_method=None)
    class1DensityCirrus = scipy.stats.gaussian_kde(class1Data['Cirrus'],bw_method=None)
    
    t_range = np.linspace(4000,20000,2000)
    fig, ax = plt.subplots()
    ax.set_color_cycle(['violet', 'blue', 'green', 'red', 'orange', 'purple', 'pink', 'black'])
    ax.set_ylim([0, 12.0e-4])
    plt.plot(t_range, class1DensityUltra_Blue(t_range))
    plt.plot(t_range, class1DensityBlue(t_range))
    plt.plot(t_range, class1DensityGreen(t_range))
    plt.plot(t_range, class1DensityRed(t_range))
    plt.plot(t_range, class1DensityNIR(t_range))
    plt.plot(t_range, class1DensitySWNIR_1(t_range))
    plt.plot(t_range, class1DensitySWNIR_2(t_range))
    plt.plot(t_range, class1DensityCirrus(t_range))
    plt.title(title)
    plt.show()


def compareClasses(originalData,band):
    temp = band
    band=band+3
    partialData = originalData.loc[originalData['Class'] == 1]
    c1DensityBand = scipy.stats.gaussian_kde(partialData.iloc[:,band],bw_method=None)
    partialData = originalData.loc[originalData['Class'] == 2]
    c2DensityBand = scipy.stats.gaussian_kde(partialData.iloc[:,band],bw_method=None)
    partialData = originalData.loc[originalData['Class'] == 3]
    c3DensityBand = scipy.stats.gaussian_kde(partialData.iloc[:,band],bw_method=None)
    partialData = originalData.loc[originalData['Class'] == 4]
    c4DensityBand = scipy.stats.gaussian_kde(partialData.iloc[:,band],bw_method=None)
    t_range = np.linspace(4000,20000,2000)
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'yellow', 'green', 'blue'])
    ax.set_ylim([0, 12.0e-4])
    plt.plot(t_range, c1DensityBand(t_range))
    plt.plot(t_range, c2DensityBand(t_range))
    plt.plot(t_range, c3DensityBand(t_range))
    plt.plot(t_range, c4DensityBand(t_range))
    plt.show()


if __name__ == '__main__':
    originalData = pd.read_csv("../Data/Training/ValidationDataImage1.csv")
    class1Data = originalData.loc[originalData['Class'] == 1]
    plotBandsPerClass(class1Data,"Class 1: Open Lands")
    class2Data = originalData.loc[originalData['Class'] == 2]
    plotBandsPerClass(class2Data,"Class 2: Settlements")
    class3Data = originalData.loc[originalData['Class'] == 3]
    plotBandsPerClass(class3Data,"Class 3: Vegetation")

    compareClasses(originalData,1)
    compareClasses(originalData,2)
    compareClasses(originalData,3)
    compareClasses(originalData,4)
    compareClasses(originalData,5)
    compareClasses(originalData,6)
    compareClasses(originalData,7)
    compareClasses(originalData,8)