from numpy import float16, loadtxt, float64, mean, std, concatenate, array
from numpy.core.fromnumeric import around

import tensorflow as tf

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import BinaryCrossentropy, MeanSquaredError, BinaryAccuracy
from keras import backend as K
from keras.optimizer_v1 import SGD 

import csv
import time
start = time.time()

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# get the dataset
def get_dataset():
    trainDataset = loadtxt("E:\\nnProject2022\Datasets\\train-data-centered.csv", delimiter=",", dtype=float16, max_rows=1000)
    testDataset = loadtxt("E:\\nnProject2022\Datasets\\test-data-centered.csv", delimiter=",", dtype=float16, max_rows=500)
    trainLabels = loadtxt("E:\\nnProject2022\RawData\\train-label.dat", delimiter=" ", dtype=int, max_rows=1000)
    testLabels = loadtxt("E:\\nnProject2022\RawData\\test-label.dat", delimiter=" ", dtype=int, max_rows=500)
    # merge datasets
    inputs = concatenate((trainDataset, testDataset), axis=0)
    targets = concatenate((trainLabels, testLabels), axis=0)
    return inputs, targets

# get the model
def get_model(n_inputs, n_outputs, n_hiddenLayerUnits, learningRate, momentumRate):
	model = Sequential()
	model.add(Dense(n_hiddenLayerUnits, input_dim=n_inputs, activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	SGD(lr=learningRate, momentum=momentumRate)
	model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[MeanSquaredError(), BinaryAccuracy()])
	return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = KFold(n_splits=5, shuffle=True)

	metricsList = [] ##

	# enumerate folds
	for i, (train_ix, test_ix) in enumerate(cv.split(X)):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]

		# define model
		model = get_model(n_inputs, n_outputs, n_inputs, 0.001, 0.01)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=20)
		# make a prediction on the test set
		yhat = model.predict(X_test)
		# round probabilities to class labels
		# yhat = around(yhat)#yhat.round()
		# calculate accuracy
		# acc = accuracy_score(y_test, yhat)
		# store result
		# print('>%.3f' % acc)
		# results.append(acc)

		# Evaluate model
		scores=model.evaluate(X_test, y_test, verbose=0)
		metricsList.append(scores)
		print("Fold :", i, " binary accuracy:", scores[2])
	Χ_prediction = model.predict(X)

	metricsList=array(metricsList)

	# create csv train dataset
	with open("E:\\nnProject2022\Datasets\\train-data-evaluation.csv","w+",newline="") as csvFile:
		csvWriter = csv.writer(csvFile,delimiter=',')
		csvWriter.writerows(Χ_prediction)

	mean_ce=mean(metricsList[:,0])
	mean_mse=mean(metricsList[:,1])
	mean_acc=mean(metricsList[:,2])
	print("mean cross entropy: ", mean_ce)
	print("mean mse: ", mean_mse)
	print("mean binary accuracy: ",mean_acc)
	

	return metricsList

# load dataset
X, y = get_dataset()
print("Finished loading dataset in: " + str(int(time.time() - start)) + " seconds")	
# evaluate model
results = evaluate_model(X, y)
print("Finished evaluating model in: " + str(int(time.time() - start)) + " seconds")	
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results[:,2]), std(results[:,2])))
print("Finished summarization in: " + str(int(time.time() - start)) + " seconds")	


    
