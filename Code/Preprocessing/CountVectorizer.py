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
vocabs.close()


#open train dataset file
trainingFile=open('E:\\nnProject2022\RawData\\train-data.dat','r')
#read each line in dataset
trainingFileArray=np.array(trainingFile.readlines())
trainingFile.close()

#Create zero val table
ZeroArrayTable=np.full((len(trainingFileArray),lenVocabs),0,dtype=int)
# ZeroArrayTable=np.full((5,lenVocabs),0,dtype=int)

activeLineNo=-1
#Fill the table
for line in trainingFileArray:
    # activeLineNo=trainingFileArray.index(line)
    activeLineNo+=1
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
trainingFileArray=np.array(trainingFile.readlines())
trainingFile.close()

#Create zero val table
ZeroArrayTable=np.full((len(trainingFileArray),lenVocabs),0,dtype=int)

activeLineNo=-1
#Fill the table
for line in trainingFileArray:
    # activeLineNo=trainingFileArray.index(line)
    activeLineNo+=1
    lineWordList=line.split()
    for word in lineWordList:
        if '>' not in word:
            #BoW implementation
            ZeroArrayTable[activeLineNo,int(word)]+=1

# create csv test dataset
with open("E:\\nnProject2022\Datasets\\test-data.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(ZeroArrayTable)
