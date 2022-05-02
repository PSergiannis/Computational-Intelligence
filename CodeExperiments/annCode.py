import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import optimizer_v1


# Read dataset 
dataset = np.loadtxt("chd.csv", delimiter=",", skiprows=(1))

# Features normalization
norm_dataset = StandardScaler().fit_transform(X=dataset)


# Split into input and output
X = norm_dataset[:, :-1]
Y = norm_dataset[:, -1]

# Split the data to training and testing data 5-Fold
kfold = KFold(n_splits=5, shuffle=True)
rmseList = []
rrseList = []

for i, (train, test) in enumerate(kfold.split(X)):
    # Create model
    model = Sequential()

    model.add(Dense(10, activation="relu", input_dim=8))
    model.add(Dense(1, activation="linear", input_dim=10))

    # Compile model
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    optimizer_v1.SGD(lr=0.08, momentum=0.2, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[rmse])

    # Fit model
    model.fit(X[train], Y[train], epochs=500, batch_size=500, verbose=0)

    # Evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    rmseList.append(scores[0])
    print("Fold :", i, " RMSE:", scores[0])

print("RMSE: ", np.mean(rmseList))

