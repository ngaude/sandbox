import pandas as pd
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def epoch_sec(t):
    #t string datetime in a 'YYYY-mm-dd HH:MM:SS' format
    # return elapsed seconds from epoch time
    dt = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    s = time.mktime(dt.timetuple())
    return s

def str_date(s):
    # s elapsed seconds from epoch time
    # return string datetime in a 'YYYY-mm-dd HH:MM:SS' format
    dt = datetime.datetime.fromtimestamp(s)
    sdt = dt.strftime('%Y-%m-%d %H:%M:%S')
    return sdt 

fname = '/home/ngaude/workspace/data/presence.csv'
df = pd.read_csv(fname, header = None)
df.columns = ['dat_heur','xmin','ymin','xmax','ymax','people']

df['hourmin'] = map(lambda d: d[11:16],df['dat_heur'])
df['we'] = df.dat_heur > '2014-05-24 04:30:00'

feature = df.groupby(['xmin','ymin','xmax','ymax','hourmin'])['people'].sum()
m = np.array(feature)
m.shape = (400,48)
m = m.astype('float')
m -= m.mean(axis=1)[:,np.newaxis]
m /= m.std(axis=1)[:,np.newaxis]

plt.figure()
plt.plot(m[100:140].transpose())
plt.title('some normalized presence')
plt.show(block=False)


pca = PCA(n_components='mle')
pca.fit(m)
pca_score = pca.explained_variance_ratio_
V = pca.components_
U = np.dot(m,V.transpose())

plt.figure()
plt.plot(np.log(pca_score))
plt.title('singular log values of presence')
plt.show(block=False)


kinertia = []
for k in range(1,100):
    km = KMeans(n_clusters = k)
    km.fit(m)
    kinertia.append(km.inertia_)

plt.figure()
plt.plot(np.log(kinertia))
plt.title('log k-inertia of k-mean of presence')
plt.show(block=False)

# choose K = 6 PCA & KMEAN confirm this model to output 6 classes.
k = 6
km = KMeans(n_clusters = k)
km.fit(m)

mp = km.labels_
mp.shape = (20,20)
plt.figure()
plt.imshow(mp)
plt.show()
