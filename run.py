from re import M
from dbscan import *
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split


# path = f'./DataSets/blobsData.csv'
path = f'/content/drive/MyDrive/Colab Notebooks/moonsData38.csv'
df = pd.read_csv(path)
data = df.values.tolist()

path = f'/content/drive/MyDrive/Colab Notebooks/moonsLabels38.csv'
df = pd.read_csv(path)
Labels = df.values.tolist()

X_train,_,Y_train,_ = train_test_split(data,Labels,test_size=0.33,random_state=42)
print(X_train)
# d = np.array(data)
d = np.array(X_train)
eps = 0.07
minpoint = 3
print(d.shape)
start_time = time.time()
#c = dbscan(d.transpose(),eps,minpoint)
c = main(d.transpose() ,eps ,minpoint )
alltime=time.time() - start_time
print(alltime)
print(c)
