import numpy as np
from pulp import *
import math
import matplotlib.pyplot as plt

"""program step:
   1. init membership U and centroids randomly
   2. update U by simplex method and calculate J
   3. update centroids with U
   4. k-means main process
"""



def loadData(filename):
    dataSet = np.mat(np.genfromtxt(filename, dtype='float64', delimiter=','))
    print(dataSet.shape)
    return dataSet

#calculate Euclidean distance
def euclDistance(vec1, vec2):
    return np.sqrt(np.sum(np.power(vec1 - vec2, 2)))

#initialize membership matrix U
def initial_U(dataSet, c):
    n, m = dataSet.shape
    U = np.mat(np.zeros((n, c)))
    count = 0
    while (count < 10):
       for j in range(n):#column列
           membershipSum = 0.0
           randoms = np.random.randint(0, 2, (n, 1))
           for i in range(c):#row行
               U[j, i] = randoms[j, 0] * (1 - membershipSum )
               membershipSum += U[j, i]
           U[j, -1] = 1 - U[j, 0]
       count = count+1
    print('initial U', U.shape)
    return U

#initialize centroids randomly
def initCentroids(dataSet, c):
    n, m = dataSet.shape
    centroids = np.mat(np.zeros((c, m)))
    count = 0
    while (count < 10):
        for j in range(m):
            minD = min(dataSet[:, j])
            maxD = max(dataSet[:, j])
            rangD = float(maxD - minD)
        for i in range(c):
            centroids[i, :] = minD + rangD * np.random.rand(c-1, 2)
        count = count+1
    print('initial_centroids:', minD, maxD)
    return centroids


#centroids with U
def calCentroids(dataSet, U, c):
    n, m= dataSet.shape
    centroids = np.mat(np.zeros((c, m)))

    for i in range(c):
        memberSum = np.sum(U[:, i])

        sampleSum = np.array(np.zeros((1, m)))
        for j in range(n):
            sampleSum += np.multiply(dataSet[j], U[j,0])

        centroids[i, :] = sampleSum / memberSum
    #print(memberSum)
    print('sample', sampleSum)
    #print('centroids:\n', centroids)
    return centroids

def upUwithSimplex(dataSet, centroids, c, k):
    n, m = dataSet.shape
    #U = np.zeros((n, c))
    J = LpProblem('cal_J', LpMinimize)
    distance1 = []
    W = []

    for j in range(n):
        for i in range(c):
            dist = np.sum(euclDistance(dataSet[j,:], centroids[i,:]))
            distance1.append(dist)
    distanceMat = np.reshape(distance1, (n, m))
    print('distance', distanceMat)
    V = {(j, i): LpVariable('V{}_{}'.format(j, i), 0, 1,LpInteger) for j in range(n) for i in range(c)}
    for i in range(c):
        for j in range(n):
            J += lpSum(distanceMat[j, i] * V[j, i])

    #add constraints
    for j in range(n):
        J += sum(V[j, i] for i in range(c)) == 1
    for i in range(c):
        J += sum(V[j, i] for j in range(n)) >= k
        J += sum(V[j, i] for j in range(n)) <= k + 1
    J.writeLP('cal_J.lp')
    J.solve()
    print('Status:', LpStatus[J.status])
    #print('Objective Function:', J)
    for v in J.variables():
        W.append(v.varValue)
        print(v.name, v.varValue)
    U = np.reshape(W, (n, m))
    #print('U:', U)
    return J, U


def ECBO(dataSet, c):
    n, m = dataSet.shape

    #1. init membership U and centroids
    #initail_U = initial_U(dataSet, c)
    initial_centroids = initCentroids(dataSet, c)
    print('initial centroids:\n', initial_centroids)

    #2. update membership U
    J, U = upUwithSimplex(dataSet, initial_centroids, c, k)

    #3. update centroids
    centroids = calCentroids(dataSet, U, c)
    print('U:\n', 'centroids:\n', 'J:\n', U, centroids, J)
    return U, centroids, J

#show cluster
def showCluster(dataSet, c, centroids, U):
    n, m = dataSet.shape

    #polt all samples
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(n):
        markIndex = int(U[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex], alpha = 0.6, )
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']

    #plot centroids
    for j in range(c):
        plt.plot(centroids[j, :], centroids[j, 1], mark[j])

    plt.show()

if __name__ == '__main__':
    dataSet = loadData('D:/Tsukuba/My Research/Program/dataset/3_circles_with_diffR/circles_with_diffR.csv')
    c = 2
    k = math.floor(len(dataSet) / c)
    U, centroids, J = ECBO(dataSet, c)#c=2
    showCluster(dataSet, c, centroids, U)
