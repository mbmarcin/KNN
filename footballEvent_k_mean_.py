import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
#import numpy as np
import matplotlib.pyplot as pl

pd.set_option('display.max_columns', 10)

# Read File
train = pd.read_csv("TrainFootballEvent.csv")

# Fill missing values
train.fillna("0", inplace=True)

# Encoding Categorical To Numerical Values
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Area'])
train['Area'] = labelEncoder.transform(train['Area'])

# labelEncoder.fit(train['Gender'])
# train['Gender'] = labelEncoder.transform(train['Gender'])

# lower age, more likely they will be interested to play
train['Age'] = train["Age"].apply(lambda x: (1 / x))

X = train.drop(['Name', 'Gender'], 1).astype(float).values

# KMeans Model
#kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=200, n_init=1, verbose=0, random_state=3425)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
#print(kmeans.labels_)
#
for i in range(0, X.shape[0]):
    if kmeans.labels_[i] == 1:
        c1 = pl.scatter(X[i, 0], X[i, 1], c='g', marker='p')
    elif kmeans.labels_[i] == 0:
        c2 = pl.scatter(X[i, 0], X[i, 1], c='r', marker='*')
#
pl.legend([c1, c2], ['Interested', 'Not Interested'])
pl.title('K-Means Of Interested Vs Not Interested Friends')
pl.show()
