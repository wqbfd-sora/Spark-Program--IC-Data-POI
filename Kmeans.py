#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
# 导入CSV格式的北京公交IC卡数据
data = pd.read_csv("Data of IC Cards in Beijing.csv", encoding="gbk")
x = data.WGS84_Lat
y = data.WGS84_Lng
plt.figure(figsize=(20, 20), dpi=200)
plt.scatter(y, x)
dataMap = data[['WGS84_Lat', 'WGS84_Lng']].values.tolist()
# print(dataMap)
# [data.WGS84_Lat,data.WGS84_Lng]

# 数据标准化
data_z = pd.DataFrame(preprocessing.StandardScaler().fit_transform(dataMap))
print(data_z)
x_z = data_z[0]
y_z = data_z[1]
plt.figure(figsize=(20, 20), dpi=200)
plt.scatter(y_z, x_z)
# 数据归一化
data_a = preprocessing.MinMaxScaler().fit(data_z)
dataa = data_a

K = range(1, 200)
# 进行聚类
meanDistortions = []
for k in K:
    jvlei = KMeans(n_clusters=k)
    jvlei.fit(data_z)
    meanDistortions.append(
        sum(
            np.min(cdist(data_z, jvlei.cluster_centers_, 'euclidean'), axis=1)
        )
    )

# 绘制碎石图
plt.plot(K, meanDistortions, 'bx--')
plt.xlabel('k')
plt.show()
