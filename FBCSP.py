


from typing import Self
import scipy.signal
from scipy import signal
import numpy as np
from DataLoader2 import DataLoader2
from pathlib import Path
import matplotlib.pyplot as plt
from mne.decoding import CSP


class FBCSP:

    def __init__(self, n_csp_components: int = 4, transform_into: str = 'csp_space'):
        self.n_csp_components = n_csp_components
        if transform_into != 'csp_space' and transform_into != 'average_power':
            raise ValueError('transform_into argument should be either csp_space or average_power')
        self.transform_into = transform_into
        self.fitted_csps: dict[CSP] | None = None

    def fit(self, data: np.array, labels: np.array, freq_rngs: list[tuple[int, int]]) -> Self:
        filtered_data = self._filter_bank(data, freq_rngs)
        self.fitted_csps = {}
        for freq_range, freq_band_data in zip(freq_rngs, filtered_data):
            fitted_csp = self._csp(freq_band_data, labels)
            self.fitted_csps[freq_range] = fitted_csp

        return self

    def transform(self, data: np.array, reshape=True) -> np.array:
        if not self.fitted_csps:
            raise RuntimeError('FBCSP has not been fitted. Call .fit() method before using .transform()')
        freq_rngs = self.fitted_csps.keys()
        n_filters = len(freq_rngs)
        filtered_data = self._filter_bank(data, freq_rngs)

        if self.transform_into == 'csp_space':
            transformed_data = np.zeros((n_filters, data.shape[-3], self.n_csp_components, data.shape[-1]))
        else:
            transformed_data = np.zeros((n_filters, data.shape[-3], self.n_csp_components))

        for i, freq_rng in enumerate(freq_rngs):
            csp: CSP = self.fitted_csps[freq_rng]
            transformed_data[i] = csp.transform(filtered_data[i])

        if reshape:
            transformed_data = np.swapaxes(transformed_data, 0, 1).reshape(len(data), -1)  # Change shape for classification convenience

        return transformed_data

    def _filter_bank(self, data: np.array, freq_rngs: list[tuple[int, int]]) -> np.array:
        n_filters = len(freq_rngs)
        filtered_data = np.zeros((n_filters, *data.shape))
        print('\nStarting data filtering...\n')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                curr_obs = data[i][j]
                for k, freq_rng in enumerate(freq_rngs):
                    b = signal.firwin(numtaps=257, cutoff=freq_rng, window=('kaiser', 8.6),
                                      pass_zero='bandpass', fs=160)
                    filtered_data[k][i][j] = scipy.signal.convolve(curr_obs, b, mode='same')

            if i % 10 == 0:
                print(f'\r{i / data.shape[0] * 100:.2f}% of data points filtered', end='', flush=True)

        print('\r100% of data points filtered\n',flush=True)
        print('Data has successfully been filtered\n', flush=True)

        return filtered_data

    def _csp(self, data: np.array, labels: np.array) -> np.array:
        csp = CSP(n_components=self.n_csp_components, transform_into=self.transform_into)
        csp.fit(data, labels)
        return csp








if __name__ == '__main__':
    data_path = Path('./epochs_tensors/imaginary')
    labels_path = data_path / 'labels.pt'
    data_loader = DataLoader2(data_path, labels_path)
    pattern = r'\d{4}_\d{2}_\d{2}.pt'
    X, y = data_loader.load(pattern=pattern)
    X, y = X[:100], y[:100]
    X = X.astype(np.float64)
    fbcsp = FBCSP(n_csp_components=4, transform_into='average_power')
    len_train = int(len(y) * 0.8)
    X_train, y_train = X[:len_train], y[:len_train]
    X_test, y_test = X[len_train:], y[len_train:]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    fbcsp.fit(X_train, y_train,[(4, 8), (8, 12), (12, 30)])
    X_transformed = fbcsp.transform(X_test, reshape=True)
    print(X_transformed.shape)


