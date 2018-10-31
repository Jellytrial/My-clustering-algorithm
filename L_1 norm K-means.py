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

#update U with minimized objective function
def updateU(dataSet, centroids, c):
    n, p = dataSet.shape
    U = np.mat(np.zeros((n, c)))
    d = np.mat(np.zeros((n, c)))
    J = 0

    """for k in range(n):
        for i in range(c):
            d[k, i] = ManhDistance(dataSet[k], centroids[i])"""


    for k in range(n):
        for i in range(c):
            for j in range(p):
                d[k, i] = ManhDistance(dataSet[k, j], centroids[i, j])
                #dist = np.min(d, axis=1)
                #return the location of min distance
                row, col = np.where(d == np.min(d, axis=1))
    #reshape row and col in to one column
    row = row.reshape(-1, 1)
    col = col.reshape(-1, 1)
    #merge row and col into a matrix
    loc = np.hstack((row, col))
    #print(loc)
    #U in the location equals 1
    for row, col in loc:
        U[row, col] = 1
    for k in range(n):
        for i in range(c):
            J = J + np.sum(U[k, i] * d[k, i])
    #print('d', d)
    print('U:\n', U)
    print('objective function:', J)
    return J, U

#update centroids
def calCentroids(dataSet, U, c):
    n, m = dataSet.shape
    centroids = np.mat(np.zeros((c, m)))
    data = np.zeros((n, m))
    #a = np.zeros((n, m), dtype=int)
    for j in range(m):
        data[:, j] = sorted(dataSet[:, j])
        a = np.argsort(dataSet.T).T
        USorted = U[a]
        USorted = USorted.reshape((n, c*m))
    #print("sorted index:", a)
    print("sorted U:", USorted)
    #print("data:", data)

    #update centroids
    S = np.zeros((c*m, 1))
    for p in range(c*m):
        S[p] = -0.5 * np.sum(USorted[:, p])
    print('S pre:', S)

    r = np.mat(np.zeros((c*m, 1), dtype=int))
    for p in range(c * m):
        while (S[p] < 0):
            r[p] = r[p] + 1
            S[p, 0] = S[p, 0] + USorted[r[p], p]
    r = r.reshape((m, c))
    print('r:', r)
    print('S after:', S)

    for i in range(c):
        for j in range(m):
            centroids[i, j] = data[r[j, i], j]

    print('new centroids: ', centroids)

    """for i in range(c):
        memberSum = np.sum(U[:, i])
        sampleSum = np.array(np.zeros((1, m)))
        for j in range(n):
            sampleSum += U[j, i]*dataSet[j]
        centroids[i] = sampleSum / memberSum
        print(memberSum)
        print('sample', sampleSum)"""
    return centroids

def L1_kMeans(dataSet, c):
    n, m = dataSet.shape
    clusterAssment = np.mat(np.zeros((n, 1)))

    initial_centroids = initCentroids(dataSet, c)
    print('initial centroids:\n', initial_centroids)

    lastJ, lastU = updateU(dataSet, initial_centroids, c)
    centroids = calCentroids(dataSet, lastU, c)
    J, U = updateU(dataSet, centroids, c)
    centroids = np.array(centroids)
    count = 0
    while count < 10:
        lastJ = J
        centroids = calCentroids(dataSet, U, c)
        J, U = updateU(dataSet, centroids, c)
        count += 1
    print('count:', count)

    for k in range(n):
        for i in range(c):
            if (U[k, i] == 1):
                clusterAssment[k] = i

    label_pred = clusterAssment
    # print(label_pred)
    CHI = calinski_harabaz_score(dataSet, label_pred)

    print('final membership:', U)
    print('objetive function:', J)
    print('final centroids:', centroids)
    #print('cluster assignment:', clusterAssment)
    print('Calinski-Harabaz Index:', CHI)
    for i in range(c):
        print(np.sum(U[:, i]))

    return U, J, centroids, clusterAssment

#show cluster
def showCluster(dataSet, centroids, c, clusterAssment):
    n, m = dataSet.shape

    mark = ['o', 'o', 'o', 'o', '^', '+', 's', 'd', '<', 'p']
    color = ['r', 'b', 'g', 'm', 'c', 'y']

    # plot centroids
    for j in range(c):
        plt.scatter(centroids[j, 0], centroids[j, 1], c='k', marker='D')

    #polt all samples
    for i in range(n):
        markIndex = int(clusterAssment[i])
        plt.scatter(dataSet[i, 0], dataSet[i, 1], marker=mark[markIndex], c=color[markIndex], alpha = 0.6)

    plt.show()

if __name__ == '__main__':
    dataSet = loadData('D:/Tsukuba/My Research/Program/dataset//3_circles_with_diffR/circles_with_diffR.csv')
    c = 2  #cluster number
    U, J, centroids, clusterAssment= L1_kMeans(dataSet, c)
    showCluster(dataSet, centroids, c, clusterAssment)
