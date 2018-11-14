import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import calinski_harabaz_score

def loadData(filename):
    dataSet = np.mat(np.genfromtxt(filename, dtype='float64', delimiter=','))
    print(dataSet.shape)
    return dataSet

#define Manhattan distance
def ManhDistance(vec1, vec2):
    return np.sum(np.abs(vec1-vec2))

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

#calculate membership U
def calUwithCent(dataSet, centroids, c, m):
    n, p = dataSet.shape
    d = np.mat(np.zeros((n, c)))
    U = np.mat(np.zeros((n, c)))
    t = 1/(m-1)

    tmp = 0

    #epsilon = np.inf

    for i in range(c):
        for k in range(n):
            d[k, i] = ManhDistance(dataSet[k, :], centroids[i, :])

    for i in range(c):
        for k in range(n):

            U[k, i] = np.sum(d[k, i]/d[k, :]) ** -t

    print('distance:', d)
    print('current U:', U)

    return U

#update centroids
def calCentwithU(dataSet, U, c, m):
    n, q = dataSet.shape
    centroids = np.mat(np.zeros((c, q)))
    data = np.zeros((n, q))
    for i in range(c):
        for k in range(n):
            U[k, i] = U[k, i] ** m
    #a = np.zeros((n, m), dtype=int)
    for j in range(q):
        data[:, j] = sorted(dataSet[:, j])
        a = np.argsort(dataSet.T).T
        USorted = U[a]
        USorted = USorted.reshape((n, c*q))
    #print("sorted index:", a)
    print("sorted U:", USorted)
    #print("data:", data)

    #update centroids
    S = np.zeros((c*q, 1))
    for p in range(c*q):
        S[p] = -0.5 * np.sum((USorted[:, p]))
    print('S pre:', S)

    r = np.mat(np.zeros((c*q, 1), dtype=int))
    for p in range(c * q):
        while (S[p] < 0):
            r[p] = r[p] + 1
            S[p, 0] = S[p, 0] + (USorted[r[p], p])
    r = r.reshape((q, c))
    print('r:', r)
    print('S after:', S)

    for i in range(c):
        for j in range(q):
            centroids[i, j] = data[r[j, i], j]

    print('new centroids: ', centroids)

    return centroids

#calculate objective function
def objFuncJ(dataSet, U, centroids, c, m):
    n, p = dataSet.shape
    d = np.mat(np.zeros((n, c)))
    J = 0

    for k in range(n):
        for i in range(c):
            d[k, i] = ManhDistance(dataSet[k], centroids[i])
            J += U[k, i] ** m * d[k, i]

    return J

def getCluster(dataSet, U, c):
    n, p = dataSet.shape
    clusterAssment = []

    clusterAssment = U.argmax(axis=1)

    """for k in range(n):
        idx, max_val =max((idx, val)for (idx, val) in enumerate(U[k, :]))
        clusterAssment.append(idx)
    #clusterAssment = np.array(clusterAssment.reshape(n, 1))"""

    print("clusterAsment:", clusterAssment)


    return clusterAssment

def fuzzyCMeans(dataSet, c, m):
    #n, p = dataSet.shape

    # 1. init centroids randomly
    initial_centroids = initCentroids(dataSet, c)

    # 2. update membership U with fixing centroids
    last_U = calUwithCent(dataSet, initial_centroids, c, m)
    last_J = objFuncJ(dataSet, last_U, initial_centroids, c, m)

    # 3. update centroids with fixing U
    centroids = calCentwithU(dataSet, last_U, c, m)

    U = calUwithCent(dataSet, centroids, c, m)
    J = objFuncJ(dataSet, U, centroids, c, m)

    count = 0
    while count < 10:
        centroids = calCentwithU(dataSet, U, c, m)
        U = calUwithCent(dataSet, centroids, c, m)
        J = objFuncJ(dataSet, U, centroids, c, m)
        count += 1

    clusterAssment = getCluster(dataSet, U, c)
    #label_pred = clusterAssment

    CHI = calinski_harabaz_score(dataSet, clusterAssment)

    print('final membership:', U)
    print('objetive function:', J)
    print('final centroids:', centroids)
    print('cluster assignment:', clusterAssment)
    print('Calinski-Harabaz Index:', CHI)
    for i in range(c):
        print("cluster size:", np.sum(U[:, i]))

    return U, centroids, J, clusterAssment

#show cluster
def showCluster(dataSet, centroids, c, clusterAssment):
    n, p = dataSet.shape

    mark = ['o', 'o', 'o', 'o', '^', '+', 's', 'd', '<', 'p']
    color = ['r', 'b', 'g', 'm', 'c', 'y']

    # plot centroids
    for j in range(c):
        plt.scatter(centroids[j, 0], centroids[j, 1], c='k', marker='D')

    # polt all samples
    for i in range(n):
        markIndex = int(clusterAssment[i])
        plt.scatter(dataSet[i, 0], dataSet[i, 1], marker=mark[markIndex], c=color[markIndex], alpha=0.6)

    plt.show()

if __name__ == '__main__':
    dataSet = loadData('D:/Tsukuba/My Research/Program/dataset//3_circles_with_diffR/circles_with_diffR.csv')
    c = 2  #cluster number
    m = 2  #fuzzy parameter
    U, centroids, J, clusterAssment = fuzzyCMeans(dataSet, c, m)
    showCluster(dataSet, centroids, c, clusterAssment)
