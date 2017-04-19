import csv
#Used for checking if classes of all the csv files given is same
i=0
with open('..\\TrainingData\\Corrected Data\\AccuracyData-2015-04-19.csv') as csvfile1:
    with open('..\\TrainingData\\Corrected Data\\AccuracyData-2015-12-31.csv') as csvfile2:
        with open('..\\TrainingData\\Corrected Data\\AccuracyData-2016-01-16.csv') as csvfile3:
            with open('..\\TrainingData\\Corrected Data\\AccuracyData-2016-03-20.csv') as csvfile4:
                spamreader1 = csv.reader(csvfile1, delimiter=',', quotechar='|')
                spamreader2 = csv.reader(csvfile2, delimiter=',', quotechar='|')
                spamreader3 = csv.reader(csvfile3, delimiter=',', quotechar='|')
                spamreader4 = csv.reader(csvfile4, delimiter=',', quotechar='|')
                for row1 in spamreader1:
                    i=i+1;
                    row2=next(spamreader2)
                    row3=next(spamreader3)
                    row4=next(spamreader4)
                    if(i!=1 and (row1[3]!=row2[3] or row1[3]!=row3[3] or
                               row1[3]!=row4[3] or row2[3]!=row3[3] or
                               row2[3]!=row4[3] or row3[3]!=row4[3])):
                        print i