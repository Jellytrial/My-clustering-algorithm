# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:37:33 2018

@author: Jelly
"""

import numpy as np
import matplotlib.pyplot as plt

#均值
mu1 = [0, 0]
mu2 = [5, 5]
mu3 = [0, 7]
mu4 = [8, 0]
#方差
cov = [[0.5,0], [0,1.5]]

#generate data
count = 0
while (count<10):
    x1,y1 = np.random.multivariate_normal(mu1,cov,80).T
    x2,y2 = np.random.multivariate_normal(mu2,cov,90).T
    x3,y3 = np.random.multivariate_normal(mu3,cov,100).T
    x4,y4 = np.random.multivariate_normal(mu4,cov,110).T
    count =count+1
x = np.concatenate([x1,x2,x3,x4])
y = np.concatenate([y1,y2,y3,y4])
#print(x)




plt.scatter(x1,y1,color='b',marker='x')
plt.scatter(x2,y2,color='b',marker='x')
plt.scatter(x3,y3,color='b',marker='x')
plt.scatter(x4,y4,color='b',marker='x')


plt.xlabel('$x$',size=20)
plt.ylabel('$y$',size=20)



plt.grid(True)
plt.legend()

#保存
plt.savefig("data1.png", format = 'png', dpi=200)
np.savetxt('data1.csv', (x,y), delimiter=',')

plt.show()
