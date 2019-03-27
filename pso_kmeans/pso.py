'''Particle Swarm Optimized Clustering
Optimizing centroid using K-Means style. 
In hybrid mode will use K-Means to seed first particle's centroid'''

import numpy as np
import matplotlib.pyplot as plt

from particle import Particle


class PSO:
    def __init__(self,
                 n_cluster: int,
                 n_particle: int,
                 data,
                 hybrid: bool = True,
                 max_iter: int = 200,
                 print_debug: int = 10):
        self.n_cluster = n_cluster
        self.n_particle = n_particle
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.hybrid = hybrid
        self.print_debug = print_debug

        self.gbest_score = np.inf
        self.gbest_centroids = None
        self.cluster = None
        self.gbest_sse = np.inf
        self._init_particle()

    def _init_particle(self):
        for i in range(self.n_particle):
            particle = None
            if i == 0 and self.hybrid:
                particle = Particle(self.n_cluster, self.data, use_kmeans=True)
            else:
                particle = Particle(self.n_cluster, self.data, use_kmeans=False)
            if particle.best_score < self.gbest_score:
                self.gbest_centroids = particle.centroids.copy()
                self.gbest_score = particle.best_score
            self.particles.append(particle)
            self.gbest_sse = min(particle.best_sse, self.gbest_sse)

    def run(self):
        print('Initial global best score', self.gbest_score)
        history = []
        for i in range(self.max_iter):
            for particle in self.particles:
                particle.update(self.gbest_centroids, self.data)
                #print(i, particle.best_score, self.gbest_score)
            for particle in self.particles:
                if particle.best_score < self.gbest_score:
                    self.gbest_centroids = particle.centroids.copy()
                    self.gbest_score = particle.best_score
            history.append(self.gbest_score)
            if i % self.print_debug == 0:
                print('Iteration {:04d}/{:04d} current gbest score {:.18f}'.format(i + 1, self.max_iter,
                                                                                   self.gbest_score))
        print('Finish with gbest score {:.18f}'.format(self.gbest_score))
        print('Fintal centroids', self.gbest_centroids)
        return history

    def show_cluter(self):
        self.cluster = self.particles.cluster
        for cent0, cent1 in self.gbest_centroids:
            plt.scatter(cent0, cent1, c='k', marker='D')
        mark = ['o', 'o', 'o', 'o', '^', '+', 's', 'd', '<', 'p']
        color = ['r', 'b', 'g', 'm', 'c', 'y']
        n, m = self.data.shape
        for i in range(n):
            markIndex = int(self.cluster[i])
            plt.scatter(self.data[i, 0], self.data[i, 1], c=color[markIndex], marker=mark[markIndex], alpha = 0.6)
        plt.show()


if __name__ == "__mian__":
    pass
