#%%all
import csv
import numpy as np
import impyute as imp
from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
data_arrays = pd.read_csv('data/data_tiroid_missing.csv')
data_arrays = data_arrays.replace('?', np.nan)
data_arrays = np.array(data_arrays, dtype=float)

data_label = np.array(data_arrays)[:, 5].tolist()

new_data = setMissingValues(data_arrays)
setMinMaxNormalization(new_data)
data_z_score = list()
data_z_score = setZscoreNormalization(new_data)
setSigmoidNormalization(data_z_score)


def count_error(data_instance, data_label):
    data_length = len(data_instance)
    knn.fit(data_instance, data_label)
    prediksi = knn.predict(data_instance)
    beda = 0

    for i, item in enumerate(data_label):
        if (item != prediksi[i]):
            beda += 1

    return (beda / data_length) * 100


#%%set_missing
def setMissingValues(data):
    data = imp.fast_knn(np.array(data)).tolist()
    error = count_error(list(np.array(data)[:, :5]),
                        list(np.array(data)[:, 5]))

    with open('data/new_tiroid.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        print("Error sebelum normalisasi ", error, "%")
        writer.writerows(data)
    csvFile.close()

    return data


#%%set_min_max
def setMinMaxNormalization(data):
    minmax_scaler = MinMaxScaler()
    data_minmax = data
    data_minmax = minmax_scaler.fit_transform(data)[:, :5].tolist()

    for i, itemi in enumerate(data_minmax):
        data_minmax[i].append(data_label[i])

    with open('data/minmax_new_tiroid.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data_minmax)
    csvFile.close()

    error = count_error(list(np.array(data_minmax)[:, :5]),
                        list(np.array(data_minmax)[:, 5]))
    print("Error normalisasi minmax ", error, "%")


#%%set_zscore
def setZscoreNormalization(data):
    data_zscore = list()
    data_zscore = data
    data_zscore = stats.zscore(data)[:, :5].tolist()

    for i, itemi in enumerate(data_zscore):
        data_zscore[i].append(data_label[i])

    with open('data/zscore_new_tiroid.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data_zscore)
    csvFile.close()

    error = count_error(list(np.array(data_zscore)[:, :5]),
                        list(np.array(data_zscore)[:, 5]))
    print("Error normalisasi zscore ", error, "%")
    return data_zscore


#%%set_sigmoid
def sigmoid(x):
    import math
    return (1 - math.exp(-x)) / (1 + math.exp(-x))


#%%set_sigmoid_normalization
def setSigmoidNormalization(data):
    data_sigmoid = list()
    data_sigmoid = data
    for i, itemi in enumerate(data):
        for j, itemj in enumerate(itemi):
            if (j < 5):
                data_sigmoid[i][j] = sigmoid(itemj)
            else:
                data_sigmoid[i][j] = itemj

    with open('data/sigmoidal_new_tiroid.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data_sigmoid)
    csvFile.close()

    error = count_error(list(np.array(data_sigmoid)[:, :5]),
                        list(np.array(data_sigmoid)[:, 5]))
    print("Error normalisasi sigmoid ", error, "%")
