import re
import numpy as np
from numpy import array
from numpy.core.multiarray import zeros 
from sklearn.feature_extraction.text import CountVectorizer
import csv

#open vocabs file
vocabs=open("E:\\nnProject2022\RawData\\vocabs.txt",'r')
#numOfRows in vocabs.dat
lenVocabs=len(vocabs.readlines())


#open train dataset file
trainingFile=open('E:\\nnProject2022\RawData\\train-data.dat','r')
#read each line in dataset
trainingFileList=trainingFile.readlines()

#Create zero val table
ZeroArrayTable=np.full((len(trainingFileList),lenVocabs),0,dtype=int)

#Fill the table
for line in trainingFileList:
    activeLineNo=trainingFileList.index(line)
    lineWordList=line.split()
    for word in lineWordList:
        if '>' not in word:
            #BoW implementation
            ZeroArrayTable[activeLineNo,int(word)]+=1

# create csv train dataset
with open("E:\\nnProject2022\Datasets\\train-data.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(ZeroArrayTable)

# ------------------------------------------------


#open test dataset file
trainingFile=open('E:\\nnProject2022\RawData\\test-data.dat','r')
#read each line in dataset
trainingFileList=trainingFile.readlines()

#Create zero val table
ZeroArrayTable=np.full((len(trainingFileList),lenVocabs),0,dtype=int)

#Fill the table
for line in trainingFileList:
    activeLineNo=trainingFileList.index(line)
    lineWordList=line.split()
    for word in lineWordList:
        if '>' not in word:
            #BoW implementation
            ZeroArrayTable[activeLineNo,int(word)]+=1

# create csv test dataset
with open("E:\\nnProject2022\Datasets\\test-data.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(ZeroArrayTable)
