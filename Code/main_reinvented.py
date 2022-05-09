import numpy as np
from numpy import float16, loadtxt, float64, mean, std, concatenate, array, nan_to_num, int32
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras.metrics import categorical_crossentropy, accuracy, CategoricalAccuracy
import keras
from keras.metrics import MeanSquaredError
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.optimizer_v1 import SGD


# Read dataset 

trainDataset = np.loadtxt("E:\\nnProject2022\Datasets\\train-data-centered.csv", delimiter=",", dtype=float16, max_rows=1000)
#testDataset = loadtxt("E:\\nnProject2022\Datasets\\test-data-centered.csv", delimiter=",", dtype=float16, max_rows=500)
trainLabels = np.loadtxt("E:\\nnProject2022\RawData\\train-label.dat", delimiter=" ", dtype=int, max_rows=1000)
#testLabels = loadtxt("E:\\nnProject2022\RawData\\test-label.dat", delimiter=" ", dtype=int, max_rows=500)

# Split into input and output
X = trainDataset
Y = trainLabels

# Split the data to training and testing data 5-Fold
kfold = KFold(n_splits=5, shuffle=True)
metricsList = []

for i, (train, test) in enumerate(kfold.split(X)):
    # Create model
    model = Sequential()
    model.add(Dense(8520, input_dim=8520, activation='relu'))
    model.add(Dense(20, input_dim=8520, activation='softmax'))
    opt =SGD(lr=0.001 , momentum=0.1)
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=[MeanSquaredError(), accuracy])

    # Fit model
    model.fit(X[train], Y[train], epochs=50, verbose=0)

    # Evaluate model
    scores=model.evaluate(trainDataset, trainLabels, verbose=0)
    metricsList.append(scores)
    print("Fold :", i, " categorical accuracy:", scores[2])

metricsList = array(metricsList)
mean_ce=mean(metricsList[:,0])
mean_mse=mean(metricsList[:,1])
mean_categ_acc=mean(metricsList[:,2])
#mean_acc=mean(metricsList[:,3])
#mean_acc_score=mean(metricsList[:,4])
print("mean cross entropy: ", mean_ce)
print("mean mse: ", mean_mse)
print("mean accuracy:",mean_categ_acc)

