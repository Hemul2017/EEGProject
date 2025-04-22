


import numpy as np
import matplotlib.pyplot as plt

random_gen = np.random.default_rng()
data1 = random_gen.normal(size=(1000, 2))
cov1 = np.array([[1, 0.4],
               [0.4, 0.7]])
data1 = data1 @ cov1
data2 = random_gen.normal(size=(1000, 2))
cov2 = np.array([[1, 0],
                 [0, 0.5]])
data2 = data2 @ cov2
plt.scatter(data1[:,0], data1[:,1])
plt.scatter(data2[:,0], data2[:,1])
plt.show()
csp = np.linalg.inv(np.cov(data2.T)) @ np.cov(data1.T)
#csp = np.linalg.inv(cov2) @ cov1
eigval, eigvec = np.linalg.eig(csp)
idx = np.argsort(eigval)[::-1]
eigval = eigval[idx]
eigvec = eigvec[:, idx]
print(eigval)
print(eigvec)
data1 = data1 @ eigvec
data2 = data2 @ eigvec
plt.scatter(data1[:,0], data1[:,1])
plt.scatter(data2[:,0], data2[:,1])
plt.show()