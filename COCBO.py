import numpy as np
from pulp import *
import math
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import calinski_harabaz_score

"""program step:
   1. init membership U or centroids randomly
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
    #print('initial U', U.shape)
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
            centroids[i, :] = minD + rangD * np.random.rand(1, m)
        count = count+1

    return centroids


#compute centroids with U
def calCentroids(dataSet, U, c):
    n, m= dataSet.shape
    centroids = np.mat(np.zeros((c, m)))
    for i in range(c):
        memberSum = np.sum(U[:, i])
        sampleSum = np.array(np.zeros((1, m)))
        for j in range(n):
            sampleSum += U[j, i]*dataSet[j]
        centroids[i] = sampleSum / memberSum
        print(memberSum)
        print('sample', sampleSum)
    #print('centroids:\n', centroids)
    return centroids

#update U by simplex method
def upUwithSimplex(dataSet, centroids, c, A1, A2):
    n, m = dataSet.shape
    d = {}
    U = []

    for j in range(n):
        for i in range(c):
            d[j,i] = euclDistance(dataSet[j], centroids[i])
    print('distance', d)

    J = LpProblem('cal_J', LpMinimize)
    #varibale
    u = {}
    for j in range(n):
        for i in range(c):
            u[j, i] = LpVariable('U_{:}_{:}'.format(j, i), 0, 1,LpInteger)

    #objective function
    J += lpSum(u[j, i] * d[j, i] for j in range(n) for i in range(c))

    #add constraints
    for j in range(n):
        J += lpSum(u[j, i] for i in range(c)) == 1
    for i in range(c):
        J += lpSum(u[j, i] for j in range(n)) >= A1
        J += lpSum(u[j, i] for j in range(n)) <= A2
    J.writeLP('cal_J.lp')
    J.solve()
    print('Status:', LpStatus[J.solve()])

    for j in range(n):
        for i in range(c):
            print(u[j, i].name)
            print(u[j, i].value())
            U.append(u[j, i].value())
    print('U:\n', U)

    U = np.reshape(U, (n, c))
    print('U_mat:\n', U)

    return J, U


def ECBO(dataSet, c):
    n, m = dataSet.shape
    clusterAssment = np.mat(np.zeros((n, 1)))

    #1. init membership  and centroids
    #initail_U = initial_U(dataSet, c)
    initial_centroids = initCentroids(dataSet, c)
    print('initial centroids:\n', initial_centroids)

    #2. update membership U
    lastJ, lastU = upUwithSimplex(dataSet, initial_centroids, c, A1, A2)
    last_centroids = calCentroids(dataSet, lastU, c)

    #3. update centroids
    J, U = upUwithSimplex(dataSet, last_centroids, c, A1, A2)
    centroids = calCentroids(dataSet, U, c)

    #while last_centroids.all() != centroids.all():
    #count = 0
    #while (count < 2):
    lastcentroids = centroids
    J, U = upUwithSimplex(dataSet, lastcentroids, c, A1, A2)
    centroids = calCentroids(dataSet, U, c)
        #count += 1

    #assgin data to clusters
    for k in range(n):
        for i in range(c):
            if (U[k, i] == 1):
                clusterAssment[k] = i
    """for i in range(n):
        minDist = 10000.0
        minIdex = 0
        for j in range(c):
            distance = euclDistance(dataSet[i], centroids[j])
            if distance < minDist:
                minDist = distance #update min distance
                minIdex = j #update assignment of data to cluster
        if clusterAssment[i, 0] != minIdex:
            clusterAssment[i, :] = minIdex"""

    label_pred = clusterAssment
    #print(label_pred)
    CHI = calinski_harabaz_score(dataSet, label_pred)
    print('final U:\n', np.mat(U))
    print('final centroids:\n', centroids)
    print('objective function:', value(J.objective))
    print('cluster assignment:', clusterAssment)
    print('Calinski-Harabaz Index:', CHI)
    for i in range(c):
        print(np.sum(U[:, i]))
    return U, centroids, J, clusterAssment

#show cluster
def showCluster(dataSet, c, centroids, clusterAssment):
    n, m = dataSet.shape

    # plot centroids
    for j in range(c):
        plt.scatter(centroids[j, 0], centroids[j, 1], c='k', marker='D', label='centroids')

    #polt all samples
    mark = ['o', 'o', 'o', 'o', '^', '+', 's', 'd', '<', 'p']
    color = ['r', 'b', 'g', 'm', 'c', 'y']
    for i in range(n):
        markIndex = int(clusterAssment[i])
        #print(markIndex)
        plt.scatter(dataSet[i, 0], dataSet[i, 1], marker=mark[markIndex], c=color[markIndex], alpha = 0.6)
        #shape = np.sum(U[:, i])
        #plt.legend((dataSet, centroids), ('cluster', 'centroids'))
    """type1=[]
    type2=[]
    type3=[]
    type4=[]
    for i in range(n):
        if clusterAssment[i, 0] == 0:
            type1.append(dataSet[i])
            n_1 = len(type1[0])
        if clusterAssment[i, 0] == 1:
            type2.append(dataSet[i])
            n_2 = len(type2[0])
        if clusterAssment[i, 0] == 2:
            type3.append(dataSet[i])
            n_3 = len(type3[0])
        if clusterAssment[i, 0] == 3:
            type4.append(dataSet[i])
            n_4 = len(type4[0])
    plt.legend((type1, type2, type3, type4), ('Cluster1(%s)'%n_1, 'Cluster2(%s)'%n_2, 'Cluster3(%s)'%n_3, 'Cluster4(%s)'%n_4))"""

    plt.show()

if __name__ == '__main__':
    dataSet = loadData('D:/Tsukuba/My Research/Program/dataset/3_circles_with_diffR/circles_with_diffR.csv')
    c = 2  #cluster number
    k = math.floor(len(dataSet) / c) #cluster size
    alpha = 28 #margin
    A1 = k - alpha
    A2 = k + alpha
    U, centroids, J, clusterAssment = ECBO(dataSet, c)
    showCluster(dataSet, c, centroids, clusterAssment)


