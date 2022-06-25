from re import M
from dbscan import *
import pandas as pd
import numpy as np
import time


# path = f'./DataSets/blobsData.csv'
path = f'/content/drive/MyDrive/Colab Notebooks/moonsData38.csv'
df = pd.read_csv(path)
data = df.values.tolist()
d = np.array(data)
eps = 0.07
minpoint = 3
print(d.shape)
start_time = time.time()
#c = dbscan(d.transpose(),eps,minpoint)
c = main(d.transpose() ,eps ,minpoint )
alltime=time.time() - start_time
print(alltime)
print(c)
