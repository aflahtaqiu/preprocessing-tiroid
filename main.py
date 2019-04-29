import csv
import numpy as np
import impyute as imp
from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

data_arrays = list()
knn = KNeighborsClassifier(n_neighbors=3)
with open('data/data_tiroid_missing.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_arrays.append(row)

data_label = np.array(data_arrays)[:, 5].tolist()

def setMissingValues(data):
    for i, itemi in enumerate(data):
        for j, itemj in enumerate(itemi):
            if(itemj == '?'):
                data[i][j] = np.nan
            else:
                data[i][j] = float (itemj)

    with open('data/new_tiroid.csv', 'w') as csvFile :
        writer = csv.writer(csvFile)
        data = imp.fast_knn(np.array(data)).tolist()
        writer.writerows(data)
    csvFile.close()

    data_nonlabel = np.array(data)[:, :5].tolist()
    print(data_nonlabel)
    data_length = len(data_nonlabel)
    knn.fit(data_nonlabel, data_label)
    prediksi = knn.predict(data_nonlabel)
    beda = 0

    for i, item in enumerate(data_label):
        if(item != prediksi[i]):
            beda = beda +1

    error = (beda/data_length)*100
    print("Error sebelum normalisasi ",error,"%")

    return data

def setMinMaxNormalization(data):
    minmax_scaler = MinMaxScaler()
    data_minmax = data
    data_minmax = minmax_scaler.fit_transform(data)[:,:5].tolist()

    for i, itemi in enumerate(data_minmax):
        data_minmax[i].append(data_label[i])

    with open('data/minmax_new_tiroid.csv', 'w') as csvFile :
        writer = csv.writer(csvFile)
        writer.writerows(data_minmax)
    csvFile.close()

    data_length = len(data_minmax)
    knn.fit(data_minmax, data_label)
    prediksi = knn.predict(data_minmax)
    beda = 0
    for i, item in enumerate(data_label):
        if(item != prediksi[i]):
            beda = beda +1
    error = (beda/data_length)*100
    print("Error normalisasi minmax ",error,"%")

def setZscoreNormalization(data) :
    data_zscore = list()
    data_zscore = data
    data_zscore = stats.zscore(data)[:,:5].tolist()

    for i, itemi in enumerate(data_zscore):
        data_zscore[i].append(data_label[i])

    with open('data/zscore_new_tiroid.csv', 'w') as csvFile :
        writer = csv.writer(csvFile)
        writer.writerows(data_zscore)
    csvFile.close()

    data_length = len(data_zscore)
    knn.fit(data_zscore, data_label)
    prediksi = knn.predict(data_zscore)
    beda = 0
    for i, item in enumerate(data_label):
        if(item != prediksi[i]):
            beda = beda +1
    error = (beda/data_length)*100
    print("Error normalisasi zscore ",error,"%")

def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))

def setSigmoidNormalization(data):
    data_sigmoid = list()
    data_sigmoid = data
    for i , itemi in enumerate(data):
        for j, itemj in enumerate(itemi):
            if(j<5):
                data_sigmoid[i][j] = sigmoid(itemj)
            else:
                data_sigmoid[i][j] = itemj

    with open('data/sigmoidal_new_tiroid.csv', 'w') as csvFile :
        writer = csv.writer(csvFile)
        writer.writerows(data_sigmoid)
    csvFile.close()

    data_length = len(data_sigmoid)
    knn.fit(data_sigmoid, data_label)
    prediksi = knn.predict(data_sigmoid)
    beda = 0
    for i, item in enumerate(data_label):
        if(item != prediksi[i]):
            beda = beda +1
    error = (beda/data_length)*100
    print("Error normalisasi sigmoid ",error,"%")


new_data = list()
new_data = setMissingValues(data_arrays)
setMinMaxNormalization(new_data)
setZscoreNormalization(new_data)
setSigmoidNormalization(new_data)