import pandas as pd
import matplotlib.pyplot as plt

from pso import PSO

if __name__ == "__main__":
    data = pd.read_csv('D:/Tsukuba/My Research/Program/dataset/1_normal_data/normal_data.csv')
    x = data.values
    pso = PSO(n_cluster=4, n_particle=10, data=x, hybrid=True)#max_iter, print_debug
    pso.run()
    pso.show_cluter()

