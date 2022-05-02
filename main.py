from numpy import loadtxt, float64, mean, std, concatenate
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

import csv
import time
start = time.time()

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

# get the dataset
def get_dataset():
    trainDataset = loadtxt("E:\\nnProject2022\Datasets\\train-data.csv", delimiter=",", dtype=int)
    testDataset = loadtxt("E:\\nnProject2022\Datasets\\test-data.csv", delimiter=",", dtype=int)
    trainLabels = loadtxt("E:\\nnProject2022\RawData\\train-label.dat", delimiter=" ", dtype=int)
    testLabels = loadtxt("E:\\nnProject2022\RawData\\test-label.dat", delimiter=" ", dtype=int)
    # merge datasets
    inputs = concatenate((trainDataset, testDataset), axis=0)
    targets = concatenate((trainLabels, testLabels), axis=0)
    return inputs, targets

# get the model
def get_model(n_inputs, n_outputs, n_hiddenLayerUnits):
	model = Sequential()
	model.add(Dense(n_hiddenLayerUnits, input_dim=n_inputs, activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[BinaryCrossentropy(), MeanSquaredError(), BinaryAccuracy()])
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
		model = get_model(n_inputs, n_outputs, n_inputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=50)
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
		metricsList.append(scores[1])
		print("Fold :", i, " binary accuracy:", scores[1])
	train_ix_prediction = model.predict(train_ix)
	# create csv train dataset
	with open("E:\\nnProject2022\Datasets\\train-data-evaluation.csv","w+",newline="") as csvFile:
		csvWriter = csv.writer(csvFile,delimiter=',')
		csvWriter.writerows(train_ix_prediction)
		
	print("mean cross entropy: ", mean(metricsList[0]))
	print("mean mse: ", mean(metricsList[1]))
	print("mean binary accuracy: ", mean(metricsList[2]))
	return metricsList

# load dataset
X, y = get_dataset()
print("Finished loading dataset in: " + str(int(time.time() - start)) + " seconds")	

# evaluate model
results = evaluate_model(X, y)
print("Finished evaluating model in: " + str(int(time.time() - start)) + " seconds")	
# summarize performance

print('Accuracy: %.3f (%.3f)' % (mean(results[2]), std(results[2])))
print("Finished summarization in: " + str(int(time.time() - start)) + " seconds")	


    
