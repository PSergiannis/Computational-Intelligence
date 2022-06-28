#calc average tf of all texts
#calc tf per text
#calc idf

import numpy as np
import csv


#open vocabs file
vocabs=open("E:\\nnProject2022\RawData\\vocabs.txt",'r')
#numOfRows in vocabs.dat
lenVocabs=len(vocabs.readlines())
vocabs.close()
print(lenVocabs)


#open train dataset file
trainingFile=open('E:\\nnProject2022\RawData\\train-data.dat','r')
#read each line in dataset
trainingFileArray=np.array(trainingFile.readlines())
lenTrainingFileArray=len(trainingFileArray)
trainingFile.close()
print(lenTrainingFileArray)


#------tf-------


#Number of apperances of each word in each text #Σε κάθε κείμενο πόσες φορές υπάρχει η κάθε λέξη
Bow = np.array(np.loadtxt("E:\\nnProject2022\Datasets\\train-data.csv", delimiter=",", dtype=np.float16))


#open test dataset file
trainingFile=open('E:\\nnProject2022\RawData\\train-data.dat','r')
#read each line in dataset
trainingFileArray=np.array(trainingFile.readlines()) #raw file
trainingFile.close()

numOfWords=np.full((lenTrainingFileArray,1),0,dtype=int)
activeLineNo=-1

for line in trainingFileArray:
    activeLineNo+=1
    lineWordList=line.split()
    for word in lineWordList:
        if '>' not in word:
            numOfWords[activeLineNo]+=1

BoW_nz=np.count_nonzero(Bow, axis=1, keepdims=True)
with open("E:\\nnProject2022\Tf-idf_tables\\Bow.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(Bow)
with open("E:\\nnProject2022\Tf-idf_tables\\Bow_nz.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(BoW_nz)

with open("E:\\nnProject2022\Tf-idf_tables\\BoW_nz.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(numOfWords)

tf=Bow
for i in range(0,lenTrainingFileArray):
    tf[i]=Bow[i]/numOfWords[i]

with open("E:\\nnProject2022\Tf-idf_tables\\tf.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(tf)


mean_tf=np.full((lenTrainingFileArray,1),0,dtype=float)
sum_tf=np.full((1,lenVocabs),0,dtype=float)
for i in range(0,lenTrainingFileArray):
    sum_tf+=tf[i]
print(sum_tf)
sum_tf=sum_tf.reshape(lenVocabs,1)

with open("E:\\nnProject2022\Tf-idf_tables\\sum_tf.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(sum_tf)

mean_tf=sum_tf/lenVocabs # 8520
print(mean_tf)
with open("E:\\nnProject2022\Tf-idf_tables\\mean_tf.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(mean_tf)





#----------Idf-----------

idf=np.full((1,lenVocabs),0,dtype=float)

for i in range(0,lenTrainingFileArray):
    for j in range(0,lenVocabs):
        if Bow[i,j]!=0:
            idf[0,j]+=1
        
idf=np.log(lenTrainingFileArray/idf)

with open("E:\\nnProject2022\Tf-idf_tables\\idf.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(idf)


#-----------tf-idf-----------

tfidf=np.full((1,lenVocabs),0,dtype=float)
for i in range(0,lenVocabs):
    tfidf[0,i]=mean_tf[i,0]*idf[0,i]

print(tfidf)

with open("E:\\nnProject2022\Tf-idf_tables\\tf-idf.csv","w+",newline="") as csvFile:
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerows(tfidf)
