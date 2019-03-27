'''k-means module'''
import numpy as np

def cal_obj_func(centroids, label, data): #objective function
    '''i: label of each cluster
       v: centroids'''
    distance = 0
    for i, v in enumerate(centroids):#index and element of centroids
        ind = np.where(label == i)
        dist = np.sum((data[ind] - v) ** 2)
        distance += dist
    return distance

class KMeans:
    '''k Means
       -----------------------
       n_cluster: int
                  Number of cluster to data
       init_flag: KmeansPlus
                  initialization method whether to use k means++ or not
                  (default is False)
       max_iter: int
                 Max iteration to update centroid
       sigma: float
              Minimum centroid update difference value to stop iteration (default: 0.0001)
       seed: int
             Seed number to use in random generator (default is None)
       centroid: list
                 List of centroids
       SSE: float
                 Sum squared error score
    '''

    def __init__(self,
                 n_cluster: int,
                 init_flag: bool = True,
                 max_iter: int = 100,
                 sigma: float = 1e-4,
                 seed: int = None):

        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.sigma = sigma
        self.init_flag = init_flag
        self.seed = seed
        self.centroid = None
        self.SSE = None

    def fit(self, data):
        '''Fit kmeans to give data
        Parameter
        ---------------------
        data: Data matrix to be fitted'''

        self.centroid = self._init_centroid(data)
        for _ in range(self.max_iter):
            distance = self._cal_distance(data)
            cluster = self._assign_cluster(distance)
            new_centroid = self._update_centroid(data, cluster)
            diff = np.abs(self.centroid - new_centroid).mean()
            self.centroid = new_centroid

            if diff <= self.sigma:
                break

        self.SSE = cal_obj_func(self.centroid, cluster, data)

    def predict(self, data):
        '''Predict new data's cluster uaing minmum distance to centroid
        Parameter
        -------------------------
        data: new data to predicted
        '''

        distance = self._cal_distance(data)
        print(distance.shape)
        cluster = self._assign_cluster(distance)
        print(cluster.shape)
        return cluster

    def _init_centroid(self, data):
        '''Initialize centroid randomly or using kmeans++
        Parameter
        ----------------
        data
        '''

        if self.init_flag:#kmeans++
            np.random.seed(self.seed)
            centroid = [int(np.random.uniform() * len(data))]
            for _ in range(1, self.n_cluster):
                dist = []
                dist = dist / dist.sum()
                cumdist = np.cumsum(dist)

                prob = np.random.rand()
                for i, c in enumerate(cumdist):
                    if prob > c and i not in centroid:
                        centroid.append(i)
                        break

            centroid = np.array([data[c] for c in centroid])
        else:#randomly initialize centroid
            np.random.seed(self.seed)
            idx = np.random.choice(range(len(data)), size=(self.n_cluster))
            centroid = data[idx]
        print("Initial centroid:", centroid)
        return centroid

    def _cal_distance(self, data):#calculate euclidean distance
        distances = []
        for c in self.centroid:
            distance = np.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = np.array(distances)
        distances = distances.T
        return distances

    def _assign_cluster(self, distance):
        '''Assign cluster to data based on minmum distance to centroid'''

        cluster = np.argmin(distance, axis=1)
        return cluster

    def _update_centroid(self, data, cluster):
        centroids = []
        for i in range(self.n_cluster):
            idx = np.where(cluster == i)
            centroid = np.mean(data[idx], axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        #print("Final centroid:", centroids)
        return centroids


if __name__ == "__main__":

    pass

