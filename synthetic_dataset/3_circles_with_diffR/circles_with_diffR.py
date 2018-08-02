import numpy as np
import matplotlib.pyplot as plt

samples_num1 = 155
samples_num2 = 100

theta1 = np.linspace(0, np.pi*2, samples_num1)
R1 = 1
x1 = np.sin(theta1)*R1
y1 = np.cos(theta1)*R1

i_set = np.arange(0, samples_num1, 1)
for i in  i_set:
    theta1 = np.random.random()*2*np.pi
    r1 = np.random.uniform(0, R1)
    x1 = np.sin(theta1)* (r1 ** 0.5)
    y1 = np.cos(theta1)* (r1 ** 0.5)
    plt.scatter(x1, y1, color='b', marker='x')
    print(x1)
    print(y1)

print("----")
theta2 = np.linspace(0, np.pi * 2, samples_num2)
R2 = 9
x2 = np.sin(theta2)*R2
y2 = np.cos(theta2)*R2

j_set = np.arange(0, samples_num2, 1)
for j in j_set:
    theta2 = np.random.random()*2* np.pi
    r2 = np.random.uniform(0, R2)
    x2 = np.sin(theta2)* (r2**0.5)+3
    y2 = np.cos(theta2)* (r2**0.5)+3
    plt.scatter(x2, y2, color='b',marker='x')
    print(x2)
    print(y2)






plt.xlim(-2, 7)
plt.ylim(-2, 7)
plt.grid(True)
plt.legend()

#save
plt.savefig("data3.png", format = 'png', dpi=200)
#np.savetxt('data3.csv', (x, y), delimiter=',')


plt.show()
