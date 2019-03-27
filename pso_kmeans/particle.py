'''Particle component for PSO'''


import numpy as np

from kmeans import KMeans, cal_obj_func


def quantization_error(centroids, label, data):
    error = 0.0
    for i, c in enumerate(centroids):
        idx = np.where(label == i)
        dist = np.linalg.norm(data[idx] - c)
        dist /= len(idx)
        error += dist
    error /= len(centroids)
    return error


class Particle:

    def __init__(self,
                 n_cluster: int,
                 data,
                 use_kmeans: bool = False,
                 w: float = 0.9,
                 c1: float = 0.5,
                 c2: float = 0.3):
        index = np.random.choice(list(range(len(data))), n_cluster)
        self.centroids = data[index].copy()
        if use_kmeans:
            kmeans = KMeans(n_cluster = n_cluster, init_flag = False)
            kmeans.fit(data)
            self.centroids = kmeans.centroid.copy()

        self.best_position = self.centroids.copy()
        self.best_score = quantization_error(self.centroids, self._predict(data), data)
        self.best_sse = cal_obj_func(self.centroids, self._predict(data), data)
        self.velocity = np.zeros_like(self.centroids)
        self._w = w
        self._c1 = c1
        self._c2 = c2

    def update(self, gbest_position, data):
        '''Update particle' velocity and centroids
        Parameter
        ----------------
        gbest_position
        data'''

        self._update_velocity(gbest_position)
        self._update_centroids(data)

    def _update_velocity(self, gbest_position):
         '''Update velocity based on previous value, 
         cognitive component, and social component'''

         v_old = self._w * self.velocity
         cognitive_component = self._c1 * np.random.random() * (self.best_position - self.centroids)
         social_component = self._c2 * np.random.random() * (gbest_position - self.centroids)
         self.velocity = v_old + cognitive_component + social_component

    def _update_centroids(self, data):
        self.centroids = self.centroids + self.velocity
        new_score = quantization_error(self.centroids, self._predict(data), data)
        sse = cal_obj_func(self.centroids, self._predict(data), data)
        self.best_sse = min(sse, self.best_sse)
        if new_score < self.best_score:
            self.best_score = new_score
            self.best_position = self.centroids.copy()

    def _predict(self, data):
        '''Predict new data's cluster using minimum distance to centroid
        '''

        distance = self._cal_distance(data)
        cluster = self._assign_cluster(distance)
        #print('cluster', cluster)
        return cluster

    def _cal_distance(self, data):
        '''Calcualate euclidean distance between datat and centroid'''

        distances = []
        for c in self.centroids:
            distance = np.sum((data - c) ** 2, axis=1)
            distances.append(distance)

        distances = np.array(distances)
        distances = np.transpose(distances)
        return distances

    def _assign_cluster(self, distance):
        cluster = np.argmin(distance, axis=1)
        return cluster


if __name__ == "__main__":

    pass
