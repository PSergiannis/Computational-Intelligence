import re
import numpy as np
from numpy import array, float64
from numpy.core.multiarray import zeros 
from sklearn.feature_extraction.text import CountVectorizer
import csv



#open vocabs file
vocabs=open("E:\\nnProject2022\RawData\\vocabs.txt",'r')
#numOfRows in vocabs.dat
lenVocabs=len(vocabs.readlines())
datasetVector=zeros(lenVocabs,dtype=float64)
datasetdatasetTransformedVector=zeros(lenVocabs,dtype=float64)


#open vectorized train dataset file
dataset = np.loadtxt("E:\\nnProject2022\Datasets\\train-data.csv", delimiter=",")
datasetLen=len(dataset)
datasetTransformed=np.full((datasetLen,lenVocabs),0,dtype=float64)
for i in range(datasetLen):
    print(i)
    datasetVector=dataset[i]
    datasetdatasetTransformedVector=datasetVector-datasetVector.mean() #Centering
    datasetTransformed[i]=datasetdatasetTransformedVector

with open("E:\\nnProject2022\Datasets\\train-data-centered.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(datasetTransformed)

# -------------------------------------------------------


#open vectorized test dataset file
dataset = np.loadtxt("E:\\nnProject2022\Datasets\\test-data.csv", delimiter=",")
datasetLen=len(dataset)
datasetTransformed=np.full((datasetLen,lenVocabs),0,dtype=float64)
for i in range(datasetLen):
    print(i)
    datasetVector=dataset[i]
    datasetdatasetTransformedVector=datasetVector-datasetVector.mean() #Centering
    datasetTransformed[i]=datasetdatasetTransformedVector

with open("E:\\nnProject2022\Datasets\\test-data-centered.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(datasetTransformed)