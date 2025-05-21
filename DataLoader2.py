import os
from pathlib import Path
import re

import numpy as np
import torch
import scipy

class DataLoader2:

    def __init__(self, data_path: Path | str, labels_path: Path | str):
        if isinstance(data_path, str):
            data_path = Path(data_path)

        if isinstance(labels_path, str):
            labels_path = Path(labels_path)

        self.data_path = data_path
        self.labels = torch.load(labels_path, weights_only=True)

    def load(self, pattern: str):
        tensor_list = []
        i = 0
        for file_name in os.listdir(self.data_path):
            if re.match(pattern, file_name):
                data = torch.load(self.data_path / file_name, weights_only=True)
                tensor_list.append(data)
                i += 1
                if i == 5:
                    break

        data = torch.stack(tensor_list)
        data = data.numpy()
        labels = self.labels
        labels = labels.numpy()

        return data, labels





    """
    sos = scipy.signal.iirfilter(8, [20,30], btype='bandpass', analog=False, ftype='butter', output='sos', fs=160)
    w, h = scipy.signal.freqz_sos(sos, worN=10000, fs=160)
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(w, db)
    plt.ylim(-75, 5)
    plt.grid(True)
    plt.yticks([0, -20, -40, -60])
    plt.ylabel('Gain [dB]')
    plt.title('Frequency Response')
    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h))
    plt.grid(True)
    plt.yticks([-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel('Phase [rad]')
    plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    plt.show()
    plt.plot(scipy.signal.sosfilt(sos, data[0][0]))
    plt.show()
    csp = CSP(transform_into='csp_space')
    csp.fit(data, labels)
    data = csp.transform(data)
    """




if __name__ == '__main__':
    data_path = Path('./epochs_tensors/imaginary')
    labels_path = data_path / 'labels.pt'
    data_loader = DataLoader2(data_path, labels_path)
    pattern = r'\d{4}_\d{2}_\d{2}.pt'
    X, y = data_loader.load(pattern=pattern)
    print(X.shape)
    print(y.shape)
