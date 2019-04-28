import csv
import numpy as np
import impyute as imp
from sklearn.preprocessing import MinMaxScaler

data_arrays = list()
with open('data/data_tiroid_missing.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_arrays.append(row)

def setMissingValues(data):
    for i, itemi in enumerate(data):
        for j, itemj in enumerate(data[i]):
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
    data_minmax = list()
    data_minmax = minmax_scaler.fit_transform(data).tolist()

    with open('data/minmax_new_tiroid.csv', 'w') as csvFile :
        writer = csv.writer(csvFile)
        writer.writerows(data_minmax)
    csvFile.close()

new_data = list()
new_data = setMissingValues(data_arrays)
setMinMaxNormalization(new_data)

