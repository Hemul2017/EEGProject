


import numpy as np
import matplotlib.pyplot as plt


class CSP:

    def __init__(self, method: str):
        if method == 'OVR' or method == 'pairwise':
            self.method = method
        else:
            raise('Wrong method name. Current methods include "OVR" and "pairwise"')

    def compute(self, data: np.array, labels:np.array) -> list[np.array]:
        computed_csp = []

        if self.method == 'OVR':
            for class_num in range(len(data)):
                one = data[labels == class_num]
                rest = data[labels != class_num]
                ovr_csp = self._csp(one, rest)
                computed_csp.append(ovr_csp)

        elif self.method == 'pairwise':
            for class_num1 in range(len(data)-1):
                for class_num2 in range(class_num1+1, len(data)):
                    data1 = data[labels == class_num1]
                    data2 = data[labels == class_num2]
                    pair_csp = self._csp(data1, data2)
                    computed_csp.append(pair_csp)

        return computed_csp




    def _csp(self, data1: np.array, data2: np.array):
        pass






if __name__ == '__main__':
    random_gen = np.random.default_rng()
    data1 = random_gen.normal(size=(1000, 2))
    cov1 = np.array([[1, 0.4],
                   [0.4, 0.7]])
    data1 = data1 @ np.linalg.cholesky(cov1).T
    data2 = random_gen.normal(size=(1000, 2))
    cov2 = np.array([[1, 0],
                     [0, 0.5]])
    data2 = data2 @ cov2
    plt.scatter(data1[:,0], data1[:,1], alpha=0.5)
    plt.scatter(data2[:,0], data2[:,1], alpha=0.5)
    plt.show()
    csp = np.linalg.inv(np.cov(data2.T)) @ np.linalg.cholesky(cov2).T
    #csp = np.linalg.inv(cov2) @ cov1
    eigval, eigvec = np.linalg.eig(csp)
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    print(eigval)
    print(eigvec)
    data1 = data1 @ eigvec
    data2 = data2 @ eigvec
    plt.scatter(data1[:,0], data1[:,1], alpha=0.5)
    plt.scatter(data2[:,0], data2[:,1], alpha=0.5)
    plt.show()