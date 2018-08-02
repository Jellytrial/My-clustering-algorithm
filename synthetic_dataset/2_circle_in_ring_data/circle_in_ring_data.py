import numpy as np
import matplotlib.pyplot as plt

samples_num1 = 150
samples_num2 = 100
t1 = np.random.random(size=samples_num1) * 2 * np.pi - np.pi#均匀分布
t2 = np.random.random(size=samples_num2) * 2 * np.pi - np.pi

#大圆环上的点
x1 = np.cos(t1) * 2
y1 = np.sin(t1) * 2
#小圆的点
x2 = np.cos(t2)
y2 = np.sin(t2)

i_set = np.arange(0, samples_num2, 1)
for i in i_set:
    len = np.sqrt(np.random.random())
    x2[i] = x2[i] * len
    y2[i] = y2[i] * len

#合并x1,x2; y1,y1
x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])



plt.scatter(x, y, color='b',marker='x')

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True)
plt.legend()

#保存
plt.savefig("data2.png", format = 'png', dpi=200)
np.savetxt('data2.csv', (x, y), delimiter=',')

plt.show()