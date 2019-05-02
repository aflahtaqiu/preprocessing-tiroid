#%%all
import csv
import numpy as np
import impyute as imp
from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut


def count_error(X_train, X_test, y_train, y_test):
    knn.fit(X_train, y_train)
    prediksi = knn.predict(X_test)
    if prediksi != y_test:
        return True

    return False


#%%set_missing
def setMissingValues(data):
    data = pd.DataFrame({
        'a': data[:, 0],
        'b': data[:, 1],
        'c': data[:, 2],
        'd': data[:, 3],
        'e': data[:, 4],
        'label': data[:, 5]
    })

    data_missing_grouped = data.groupby('label')

    new_data_grouped = list()
    for key, item in data_missing_grouped:
        temp = list(imp.fast_knn(np.array(item), k=3))
        for i in temp:
            new_data_grouped.append(i)

    with open('data/new_tiroid.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(new_data_grouped)
    csvFile.close()

    return new_data_grouped


#%%set_min_max
def setMinMaxNormalization(data):
    minmax_scaler = MinMaxScaler()

    X = np.array(data)[:, :5]
    y = np.array(data)[:, 5]

    loo = LeaveOneOut()
    loo.get_n_splits(X)

    error = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = list(minmax_scaler.fit_transform(X_train))
        X_test = minmax_scaler.transform(X_test)
        if (count_error(X_train, X_test, y_train, y_test)):
            error += 1

    print('Error Min-Max : ', (error / len(data)) * 100, '%')


#%%set_zscore
def setZscoreNormalization(data):
    X = np.array(data)[:, :5]
    y = np.array(data)[:, 5]

    loo = LeaveOneOut()
    loo.get_n_splits(X)

    error = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for i in range(0, len(X_train[0])):
            X_test[0, i] = (X_test[0, i] - stats.tmean(
                X_train[:, i])) / stats.tstd(X_train[:, i])
        X_train = list(stats.zscore(X_train))
        if (count_error(X_train, X_test, y_train, y_test)):
            error += 1

    print('Error Z-Score : ', (error / len(data)) * 100, '%')


#%%set_sigmoid
def sigmoid(x):
    import math
    return (1 - math.exp(-x)) / (1 + math.exp(-x))


#%%set_sigmoid_normalization
def setSigmoidNormalization(data):
    X = np.array(data)[:, :5]
    y = np.array(data)[:, 5]

    loo = LeaveOneOut()
    loo.get_n_splits(X)

    error = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for i in range(0, len(X_train[0])):
            X_test[0, i] = sigmoid(
                (X_test[0, i] - stats.tmean(X_train[:, i])) /
                stats.tstd(X_train[:, i]))
        X_train = [[sigmoid(itemj) for itemj in item]
                   for item in stats.zscore(X_train)]

        if (count_error(X_train, X_test, y_train, y_test)):
            error += 1

    print('Error Sigmoid : ', (error / len(data)) * 100, '%')


knn = KNeighborsClassifier(n_neighbors=3)
data_arrays = pd.read_csv('data/data_tiroid_missing.csv')
data_arrays = data_arrays.replace('?', np.nan)
data_arrays = np.array(data_arrays, dtype=float)

data_label = np.array(data_arrays)[:, 5].tolist()

new_data = setMissingValues(data_arrays)
setMinMaxNormalization(new_data)
setZscoreNormalization(new_data)
setSigmoidNormalization(new_data)