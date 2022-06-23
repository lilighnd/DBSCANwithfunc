from re import M
from dbscan import *
import pandas as pd
import numpy as np
import time
from sklearn import cluster, datasets

# blobs = datasets.make_blobs(n_samples=25000, n_features = 2, centers = 3)
# data=blobs[0]
start_time = time.time()
# path = f'./blobs/blobsData1m.csv'
path = f'/content/drive/MyDrive/Colab Notebooks/blobsData1m.csv'
df = pd.read_csv(path)
data = df.values.tolist()
d = np.array(data)
eps = 1
minpoint = 5
print(d.shape)
#c = dbscan(d.transpose(),eps,minpoint)
c = main(d.transpose() ,eps ,minpoint )
print(c)
alltime=time.time() - start_time
print(alltime)