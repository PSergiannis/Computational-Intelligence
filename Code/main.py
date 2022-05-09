from keras.optimizer_v1 import SGD
from numpy import float16, loadtxt, float64, mean, std, concatenate, array, nan_to_num, int32
from numpy.core.fromnumeric import around

import tensorflow as tf

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import MeanSquaredError, BinaryAccuracy, accuracy, categorical_accuracy
from keras import backend as K
from keras.optimizer_v1 import SGD

import csv
import time
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()
start = time.time()

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#get the datasets
def get_datasets():
    trainDataset = loadtxt("E:\\nnProject2022\Datasets\\train-data-normalized.csv", delimiter=",", dtype=float16)
    testDataset = loadtxt("E:\\nnProject2022\Datasets\\test-data-normalized.csv", delimiter=",", dtype=float16)
    trainLabels = loadtxt("E:\\nnProject2022\RawData\\train-label.dat", delimiter=" ", dtype=int)
    testLabels = loadtxt("E:\\nnProject2022\RawData\\test-label.dat", delimiter=" ", dtype=int)
	
    # merge datasets
	#inputs = concatenate((trainDataset, testDataset), axis=0) #
	#targets = concatenate((trainLabels, testLabels), axis=0) #
    X_train = trainDataset
    Y_train = trainLabels
    X_test = testDataset
    Y_test = testLabels
    return X_train, Y_train, X_test, Y_test


# get the model
def get_model(n_inputs, n_outputs, n_hiddenLayerUnits, learningRate, momentumRate):
	model = Sequential()
	model.add(Dense(n_hiddenLayerUnits, input_dim=n_inputs, activation='relu'))
	model.add(Dense(n_outputs, input_dim=n_hiddenLayerUnits, activation='sigmoid'))
	opt=SGD(lr=learningRate, momentum=momentumRate)
	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[MeanSquaredError(), categorical_accuracy])
	return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X_train, Y_train, X_test, Y_test):
	results = list()
	n_inputs, n_outputs = X_train.shape[1], Y_train.shape[1]
	# define evaluation procedure
	cv = KFold(n_splits=5, shuffle=True)

	# trainHistoryList = []
	testCVFoldMetricsList = []

	# enumerate folds
	for i, (train_ix, test_ix) in enumerate(cv.split(X_train)):
		# prepare data
		x_cv_train, x_cv_test = X_train[train_ix], X_train[test_ix]
		y_cv_train, y_cv_test = Y_train[train_ix], Y_train[test_ix]

		# define model
		model = get_model(n_inputs, n_outputs, n_inputs, 0.01, 0.01)

		# fit model
		activeTrainingHistory= model.fit(x_cv_train, y_cv_train, verbose=0, epochs=20)
		# trainHistoryList.append(activeTrainingHistory)

		# Evaluate model
		testActiveFoldMetrics=model.evaluate(x_cv_test, y_cv_test, verbose=0)
		testCVFoldMetricsList.append(testActiveFoldMetrics)
		print("Fold :", i, " CV-test evaluation accuracy:", testActiveFoldMetrics[2])

	testCVFoldsMetrics = array(testCVFoldMetricsList)
	
	# Evaluate testdataset
	testDatasetEvaluationMetrics=model.evaluate(X_test, Y_test, verbose=0)

	# create csv test predictions
	# testPrediction = around(model.predict(testDataset))
	# with open("E:\\nnProject2022\Datasets\\test-data-prediction.csv","w+",newline="") as csvFile:
	# 	csvWriter = csv.writer(csvFile,delimiter=',')
	# 	csvWriter.writerows(testPrediction)


	# return the metrics of repeated CV, last model fit, evaluation metrics of test dataset
	return testCVFoldsMetrics, activeTrainingHistory, testDatasetEvaluationMetrics

def plot_result(item):
    plt.plot(activeTrainingHistory.history[item], label=item)
    # plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

# load dataset
X_train, Y_train, X_test, Y_test = get_datasets()
print("Finished loading dataset in: " + str(int(time.time() - start)) + " seconds")	
# evaluate model
testCVFoldsMetrics, activeTrainingHistory, testDatasetEvaluationMetrics = evaluate_model(X_train, Y_train, X_test, Y_test)
# metrics, history = evaluate_model(X, y)

mean_ce=mean(testCVFoldsMetrics[:,0])
mean_mse=mean(testCVFoldsMetrics[:,1])
mean_acc=mean(testCVFoldsMetrics[:,2])
print("mean cross entropy: ", mean_ce)
print("mean mse: ", mean_mse)
print("mean categorical accuracy:",mean_acc)

plot_result("loss")
plot_result("mean_squared_error")
plot_result("categorical_accuracy")

print("Finished evaluating model in: " + str(int(time.time() - start)) + " seconds")	
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(testCVFoldsMetrics[:,2]), std(testCVFoldsMetrics[:,2])))
print("Finished summarization in: " + str(int(time.time() - start)) + " seconds")	


    
