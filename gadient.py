import pandas as pd
import numpy as np
import sys
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
label_encoder = LabelEncoder()


def gradientDescent(X, y, theta, learning_rate, num_iters):
    m = len(y)

    for i in range(1, num_iters):
        hypothesis = np.dot(X, theta)
        errors = hypothesis - y

        newDecrement = (learning_rate * (1/m) * np.dot(np.transpose(errors), X))
        theta = theta - np.transpose(newDecrement)

    return theta


def featureNormalize(X, n):
    X_norm = X
    mu = np.zeros(n, dtype=np.int)
    sigma = [0] * n

    for i in range(0, n):
        meanXi = np.mean(X[:, i])
        mu[i] = meanXi

        X_norm[:, i] = [x - mu[i] for x in X_norm[:, i]]

        standardXi = np.std(X[:, i])
        sigma[i] = standardXi

        X_norm[:, i] = [x / sigma[i] for x in X_norm[:, i]]

    return X_norm, mu, sigma


def run():
    # load data
    raw_data1 = pd.read_csv('student_train.csv')
    raw_data1.head()
    raw_data1.dropna()
    y = raw_data1.iloc[:, -1]
    raw_data = raw_data1.iloc[:, 0:17]
    
    for column in raw_data:
        raw_data[column] = label_encoder.fit_transform(raw_data[column])
        raw_data[column].unique()
    n = 17
    X = raw_data.to_numpy()
    print(X)
    y = y.to_numpy()
    m = len(y)
    X, mu, sigma = featureNormalize(X, n)

    ones = np.ones(m, dtype=np.int)
    X = np.column_stack((ones, X))
    learning_rate = 0.01
    num_iters = 500
    theta = np.zeros((n+1, 1), dtype=np.int)
    
    theta = gradientDescent(X, y, theta, learning_rate, num_iters)
    test = pd.read_csv('student_test.csv')
    test.dropna()
    raw_test = test.iloc[:, 0:17]
    ytest = test.iloc[:, -1]
    for column in raw_test:
        unique_vals = np.unique(raw_test[column])
        raw_test[column] = label_encoder.fit_transform(raw_test[column])
        raw_test[column].unique()
    n = 17
    Y = []
    mse =[]
    Xtest = raw_test.to_numpy()
    ytest = ytest.to_numpy()
    for i in range(0, n):
        Xtest[:, i] = [x - mu[i] for x in Xtest[:, i]]
        Xtest[:, i] = [x / sigma[i] for x in Xtest[:, i]]
    for i in range(0, len(Xtest)):
        Y = [1] + Xtest[i].tolist()
        Result = np.dot(Y, theta)
        print(Result[0])
        mse.append(Result[0])
        Y = []
    print(mse)
    print(ytest)
    print(mean_squared_error(ytest, mse))

if __name__ == '__main__':
    run()
