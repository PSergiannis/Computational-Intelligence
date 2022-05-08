import re
import numpy as np
from numpy import array, float64
from numpy.core.multiarray import zeros 
from sklearn.feature_extraction.text import CountVectorizer
import csv
from sklearn import preprocessing


center_function = lambda x: x - x.mean()
# open vectorized train dataset csv
dataset = np.loadtxt("E:\\nnProject2022\Datasets\\train-data.csv", delimiter=",")
datasetTransformed = center_function(dataset) 
with open("E:\\nnProject2022\Datasets\\train-data-centered.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(datasetTransformed)

# --------------------------------------------------

# open vectorized test dataset csv
dataset = np.loadtxt("E:\\nnProject2022\Datasets\\test-data.csv", delimiter=",")
datasetTransformed = center_function(dataset) 
with open("E:\\nnProject2022\Datasets\\test-data-centered.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(datasetTransformed)

 