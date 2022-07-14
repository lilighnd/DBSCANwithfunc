from re import M
from dbscan import *
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import cluster, datasets
import json
from sklearn.metrics import adjusted_rand_score

# # path = f'./DataSets/blobsData.csv'
# path = f'/content/drive/MyDrive/Colab Notebooks/moonsData38.csv'
# df = pd.read_csv(path)
# data = df.values.tolist()

# path = f'/content/drive/MyDrive/Colab Notebooks/moonsLabels38.csv'
# df = pd.read_csv(path)
# Labels = df.values.tolist()

# X_train,X_test,Y_train,Y_test=train_test_split(data,Labels,test_size=0.33,random_state=42)
# print(X_test)
# # d = np.array(data)
# d = np.array(X_test)

# eps = 0.07
# minpoint = 3
Dataset = "blob"
Number_Data = 50
Noise = 0.05
Random_state = 42
features = 2
Centers = 3
Epsilon = 0.01
Minpoints = 3
argumentList = sys.argv[1:]
# Options
options = "d:n:N:r:f:c:e:m:"
 
# Long options
long_options = ["Dataset", "Number_Data", "Noise", "Random_state", "features", "Centers", "Epsilon", "Minpoints"]

arguments, values = getopt.getopt(argumentList, options, long_options)

for currentArgument, currentValue in arguments:

    if currentArgument in ("-d", "--Dataset"):
        Dataset = currentValue

    elif currentArgument in ("-n", "--Number_Data"):
        Number_Data = currentValue

    elif currentArgument in ("-N", "--Noise"):
        Noise = currentValue

    elif currentArgument in ("-r", "--Random_state"):
        Random_state = currentValue
            
    elif currentArgument in ("-f", "--features"):
        features = currentValue

    elif currentArgument in ("-c", "--Centers"):
        Centers = currentValue

    elif currentArgument in ("-e", "--Epsilon"):
        Epsilon = currentValue
    
    elif currentArgument in ("-m", "--Minpoints"):
        Minpoints = currentValue

Obj = {

    "data" : Dataset,
    "n_samples" : Number_Data,
    "noise" : Noise,
    "random_state" : Random_state,
    "features" : features,
    "centers" : Centers,
    "Eps" : Epsilon,
    "Minpts" : Minpoints,
}
json_object = json.dumps(Obj, indent = 9)
with open("/content/drive/MyDrive/Colab Notebooks/inputobjectdb.json", "w") as outfile:
    outfile.write(json_object)

with open('/content/drive/MyDrive/Colab Notebooks/inputobjectdb.json', 'r') as openfile:
  
    # Reading from json file
    json_object = json.load(openfile)








with open('/content/drive/MyDrive/Colab Notebooks/inputobjectdb.json', 'r') as openfile:
# Reading from json file
    json_object = json.load(openfile)

D = json_object["data"]
Numbers = int(json_object["n_samples"])
Noise = int(json_object["noise"])
R = int(json_object["random_state"])
f = int(json_object["features"])
c = int(json_object["centers"])


True_label=[]
if D == "moon":
    print("moons data")       
    data = datasets.make_moons(n_samples=Numbers,noise=Noise,random_state=R)

if D == "blob":
    print("blobs data")       
    data = datasets.make_blobs(n_samples=Numbers, n_features = f, 
            centers = 3,cluster_std = 0,random_state=R)

if D == "circle":
    print("circles data")       
    data = datasets.make_circles(n_samples=Numbers,noise=Noise,random_state=R,factor=0.8)


True_label = data[1]
# Data=data[0].tolist()
Data=data[0]
print(f"Labels : {True_label}")
print(f"Data : {Data}")


start_time = time.time()
#c = dbscan(d.transpose(),eps,minpoint)
c = main(Data ,Epsilon, Minpoints)
alltime=time.time() - start_time
R1 = adjusted_rand_score(True_label, c)
print(R1,alltime)
print(c)
