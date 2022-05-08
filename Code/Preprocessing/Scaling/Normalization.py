import re
import numpy as np
from numpy import array, float64
from numpy.core.multiarray import zeros 
from sklearn.feature_extraction.text import CountVectorizer
import csv
from sklearn import preprocessing


scaler = preprocessing.MinMaxScaler()
# open vectorized train dataset csv
dataset = np.loadtxt("E:\\nnProject2022\Datasets\\train-data.csv", delimiter=",")
datasetTransformed = scaler.fit_transform(dataset) 
with open("E:\\nnProject2022\Datasets\\train-data-normalized.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(datasetTransformed)

# --------------------------------------------------

# open vectorized test dataset csv
dataset = np.loadtxt("E:\\nnProject2022\Datasets\\test-data.csv", delimiter=",")
datasetTransformed = scaler.fit_transform(dataset) 
with open("E:\\nnProject2022\Datasets\\test-data-normalized.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(datasetTransformed)

 