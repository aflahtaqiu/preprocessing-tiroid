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

    return data

def setMinMaxNormalization(data):
    minmax_scaler = MinMaxScaler()
    data_minmax = data
    data_label = np.array(data)[:, 5].tolist()
    data_minmax = minmax_scaler.fit_transform(data)[:,:5].tolist()
    
    knn.fit(data_arrays, data_label)

    for i, itemi in enumerate(data_minmax):
        data_minmax[i].append(data_label[i])

    hasil = knn.predict(data_minmax)
    print(hasil)

    with open('data/minmax_new_tiroid.csv', 'w') as csvFile :
        writer = csv.writer(csvFile)
        writer.writerows(data_minmax)
    csvFile.close()

def setZscoreNormalization(data) :
    data_zscore = list()
    data_zscore = data
    data_label = np.array(data)[:, 5].tolist()
    data_zscore = stats.zscore(data)[:,:5].tolist()

    for i, itemi in enumerate(data_zscore):
        data_zscore[i].append(data_label[i])

    with open('data/zscore_new_tiroid.csv', 'w') as csvFile :
        writer = csv.writer(csvFile)
        writer.writerows(data_zscore)
    csvFile.close()

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


new_data = list()
new_data = setMissingValues(data_arrays)
setMinMaxNormalization(new_data)
setZscoreNormalization(new_data)
setSigmoidNormalization(new_data)