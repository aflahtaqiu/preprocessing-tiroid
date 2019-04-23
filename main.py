import csv
import numpy as np
import impyute as imp

data_arrays = list()
results = []
with open('data/data_tiroid_missing.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        data_arrays.append(row)

for i , itemi in enumerate(data_arrays):
    for j, itemj in enumerate(data_arrays[i]):
        if(itemj == '?'):
            data_arrays[i][j] = np.nan
        else:   
            data_arrays[i][j] = float(itemj)

with open('data/new_tiroid.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(imp.fast_knn(np.array(data_arrays)).tolist())
csvFile.close()
