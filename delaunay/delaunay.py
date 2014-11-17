
###################################################
# this script produces a neighboring list of iris #
# based on a delaunay triangulation               #
###################################################

import numpy as np
from scipy.spatial import Delaunay

# input file "iris_xy_csv" csv file with code_iris,centroid_x,centroid_y
f=open('iris_xy.csv','r')



# skip csv header
f.readline()

points_array = []
iris_array = []

for line in f:
    elt = line.replace('"','').split(',')
    iris = elt[0]
    x = float(elt[1])
    y = float(elt[2])
    points_array.append([x, y])
    iris_array.append(iris)

f.close()

points = np.asarray(points_array)
tri = Delaunay(points)

import matplotlib.pyplot as plt
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
#plt.plot(points[:,0], points[:,1], 'o')
plt.show()


neighbors = []

# first element are elements themselves e.g zero order neighbors
zero = [set((i,)) for i in range(len(iris_array))]
neighbors.append( zero )

# second element are first order neighbors extracted from delaunay triangulation
first = [[] for i in range(len(iris_array))]
for t in tri.simplices:
    first[t[0]].append(t[1])
    first[t[0]].append(t[2])
    first[t[1]].append(t[0])
    first[t[1]].append(t[2])
    first[t[2]].append(t[0])
    first[t[2]].append(t[1])

first=map(set,first)
neighbors.append( first )

# next elements are next order neighbors

def nextNeighbors(n):
    # assume n contains zero and first order neighbors :
    # append to n next order neighbors
    nxt = [set() for i in range(len(n[0]))]
    for i,s in enumerate(n[-1]):
        for j in s:
            # union all know first-order neighbors of "i"
            nxt[i] |= n[1][j]
        # remove previous neighbors to "i" and "i" itself
        nxt[i] -= n[-1][i]
        nxt[i] -= n[-2][i]
    n.append(nxt)
    return

def sizeNeighbors(n):
    size = 0
    for o in n:
        for s in o:
            size += len(s)
    return size    

def displayNeighbors(n):
    print("code_iris,neighbor,rank") 
    # for each order
    for o,c in enumerate(n):
        # for each iris
        for i,s in enumerate(c):
            # for each neighbors
            for j in s:
                print(iris_array[i]+","+iris_array[j]+","+str(o)) 
    return



# 2nd to 5th order neighbors
nextNeighbors(neighbors) #2nd
nextNeighbors(neighbors) #3rd
nextNeighbors(neighbors) #4th
nextNeighbors(neighbors) #5th

#dump results
displayNeighbors(neighbors)

