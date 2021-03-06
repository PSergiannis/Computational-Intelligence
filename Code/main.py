from keras import optimizer_v1
from keras import optimizers
from keras.optimizer_experimental import optimizer
from keras.optimizer_v1 import SGD, Optimizer
from numpy import float16, loadtxt, float64, mean, std, concatenate, array, nan_to_num, int32
from numpy.core.fromnumeric import around

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import MeanSquaredError, BinaryAccuracy, accuracy, categorical_accuracy
from keras import backend as K
#from keras.optimizer_v1 import Adam, SGD
#from keras.optimizers import gradient_descent_v2.SGD

import csv
import time
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
start = time.time()

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
	model.add(Dense(n_hiddenLayerUnits, input_dim=n_inputs, activation='relu')) #1st hidden layer
	#model.add(Dense(500, input_dim=n_hiddenLayerUnits, activation='relu')) #2nd hidden layer
	model.add(Dense(n_outputs, input_dim=n_hiddenLayerUnits, activation='sigmoid')) #outputs
	opt = tf.keras.optimizers.SGD(learning_rate=learningRate, momentum=momentumRate, nesterov=False, name="SGD")
	#opt = Adam(lr=0.001)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[MeanSquaredError(), categorical_accuracy])
	return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X_train, Y_train, X_test, Y_test):
	results = list()
	n_inputs, n_outputs = X_train.shape[1], Y_train.shape[1]
	# define evaluation procedure
	cv = KFold(n_splits=5, shuffle=True)

	# trainHistoryList = []
	testCVFoldMetricsList = []
	testDatasetEvaluationMetricsList = []
	# enumerate folds
	for i, (train_ix, test_ix) in enumerate(cv.split(X_train)):
		# prepare data
		x_cv_train, x_cv_test = X_train[train_ix], X_train[test_ix]
		y_cv_train, y_cv_test = Y_train[train_ix], Y_train[test_ix]

		# define model
		model = get_model(n_inputs, n_outputs, (n_inputs+n_outputs)/2 , 0.1, 0.06)

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
	testDatasetEvaluationMetricsList.append(testDatasetEvaluationMetrics)
	testDatasetEvaluationMetrics = array(testDatasetEvaluationMetricsList)
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

cv_mean_ce=mean(testCVFoldsMetrics[:,0])
cv_mean_mse=mean(testCVFoldsMetrics[:,1])
cv_mean_acc=mean(testCVFoldsMetrics[:,2])
print("---Train---")
print("cv mean cross entropy: ", cv_mean_ce)
print("cv mean mse: ", cv_mean_mse)
print("cv mean categorical accuracy:",cv_mean_acc)

test_mean_ce=mean(testDatasetEvaluationMetrics[:,0])
test_mean_mse=mean(testDatasetEvaluationMetrics[:,1])
test_mean_acc=mean(testDatasetEvaluationMetrics[:,2])
print("---Test---")
print("test mean cross entropy: ", test_mean_ce)
print("test mean mse: ", test_mean_mse)
print("test mean categorical accuracy:",test_mean_acc)

plot_result("loss")
plot_result("mean_squared_error")
plot_result("categorical_accuracy")

print("Finished evaluating model in: " + str(int(time.time() - start)) + " seconds")	
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(testCVFoldsMetrics[:,2]), std(testCVFoldsMetrics[:,2])))
print("Finished summarization in: " + str(int(time.time() - start)) + " seconds")	


    
