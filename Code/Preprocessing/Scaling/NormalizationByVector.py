import re
import numpy as np
from numpy import array, float64
from numpy.core.multiarray import zeros 
from sklearn.feature_extraction.text import CountVectorizer
import csv


#     a = 0
#     b = 1
#     tempVector = a + ((tempVector - np.min(tempVector)) * (b - a) / (np.max(tempVector) - np.min(tempVector)))#Normalization


#open vocabs file
vocabs=open("E:\\nnProject2022\RawData\\vocabs.txt",'r')
#numOfRows in vocabs.dat
lenVocabs=len(vocabs.readlines())
datasetVector=zeros(lenVocabs,dtype=float64)
datasetdatasetTransformedVector=zeros(lenVocabs,dtype=float64)
# limits for normalization
a = 0
b = 1

# open vectorized train dataset csv
dataset = np.loadtxt("E:\\nnProject2022\Datasets\\train-data.csv", delimiter=",")
datasetLen=len(dataset)
datasetTransformed=np.full((datasetLen,lenVocabs),0,dtype=float64)

for i in range(datasetLen):
    print(i)
    datasetVector=dataset[i]
    datasetdatasetTransformedVector = a + ((datasetVector - np.min(datasetVector)) * (b - a) / (np.max(datasetVector) - np.min(datasetVector)))#Normalization
    #tempVector = a + ((tempVector - np.min(tempVector)) * (b - a) / (np.max(tempVector) - np.min(tempVector)))#Normalization
    datasetTransformed[i]=datasetdatasetTransformedVector

with open("E:\\nnProject2022\Datasets\\train-data-normalizedByVector.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(datasetTransformed)

# --------------------------------------------------

# open vectorized test dataset csv
dataset = np.loadtxt("E:\\nnProject2022\Datasets\\test-data.csv", delimiter=",")
datasetLen=len(dataset)
datasetTransformed=np.full((datasetLen,lenVocabs),0,dtype=float64)

for i in range(datasetLen):
    print(i)
    datasetVector=dataset[i]
    datasetdatasetTransformedVector = a + ((datasetVector - np.min(datasetVector)) * (b - a) / (np.max(datasetVector) - np.min(datasetVector)))#Normalization
    #tempVector = a + ((tempVector - np.min(tempVector)) * (b - a) / (np.max(tempVector) - np.min(tempVector)))#Normalization
    datasetTransformed[i]=datasetdatasetTransformedVector

with open("E:\\nnProject2022\Datasets\\test-data-normalizedByVector.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(datasetTransformed)

 