


import numpy as np
import pandas as pd
import mne_connectivity
from DataLoader import EEGDataLoader
from pathlib import Path
import torch



class EEGDataSaver:

    def __init__(self, load_path: Path | str, pattern: str):
        self.load_path = load_path
        self.pattern = pattern
        self.first_type_runs = {'03', '04', '07', '08', '11', '12'}
        self.encode_dict = {'rest': 0, 'left_hand': 1, 'right_hand': 2, 'both_hands': 3, 'feet': 4}

    def save_epochs(self, save_path: Path | str,
                    connectivity: bool = False,
                    connectivity_method: str | None = None,
                    connectivity_freqs: list[float] | None = None,
                    **kwargs) -> None:

        if isinstance(save_path, str):
            save_path = Path(save_path)

        data_loader = EEGDataLoader(self.load_path, self.pattern, **kwargs)

        if connectivity:
            n_samples_expected = 4096
        else:
            n_samples_expected = 657

        for file_idx, (epochs, file_name) in enumerate(data_loader):
            participant_id = file_name[1:4]
            n_run = file_name.split('.')[0][-2:]
            # Пропуск базовых сессий
            if (n_run == '01') or (n_run == '02'):
                continue
            # Разделение данных на представляемые и реальные движения
            if int(n_run) % 2 == 0:
                data_type = 'i'
            else:
                data_type = 'r'

            for epoch_idx, (evoked, epoch_label) in enumerate(
                    zip(epochs.iter_evoked(), epochs.get_annotations_per_epoch())):
                print(file_name)
                data = evoked.get_data()

                if connectivity:
                    data = data[np.newaxis, :, :]
                    data = mne_connectivity.spectral_connectivity_time(data, freqs=connectivity_freqs, sfreq=160,
                                                                       method=connectivity_method)
                    data = np.squeeze(data.get_data()).T

                if data.shape[1] != n_samples_expected:
                    print(
                        f'Record {file_idx:04}_{epoch_idx:02} has {data.shape[1]} samples, while {n_samples_expected} samples expected')
                    continue
                data_tensor = torch.tensor(data, dtype=torch.float32)
                save_path.mkdir(parents=True, exist_ok=True)
                label = self._encode_labels(epoch_label[0][2], n_run)
                save_file_name = f'{data_type}_{participant_id}_{n_run}_{epoch_idx:02}_{label}.pt'
                torch.save(data_tensor, save_path / save_file_name)

            print(f'\nFile {file_name} has successfully been processed\n')

        print(f'\nAll files have successfully been processed\n')


    def _encode_labels(self, annotation_label: str, n_run: str) -> int:

        if annotation_label == 'T0':
            return self.encode_dict['rest']
        elif annotation_label == 'T1':
            if n_run in self.first_type_runs:
                return self.encode_dict['left_hand']
            else:
                return self.encode_dict['both_hands']
        else:
            if n_run in self.first_type_runs:
                return self.encode_dict['right_hand']
            else:
                return self.encode_dict['feet']


if __name__ == '__main__':
    data_saver = EEGDataSaver(load_path='./files', pattern=r'\w\d{3}')
    #data_saver.save_to_2d_tensors_by_epochs('./epochs_tensors', baseline=None, tmin=0, tmax=4.1)
    #data_saver.save_to_2d_tensors_by_epochs_wo_division('./epochs_tensors_wo_division', baseline=None, tmin=0, tmax=4.1)
    #data_saver.save_epochs('./epochs_tensors', connectivity=False, baseline=None, tmin=0, tmax=4.1)
    data_saver.save_epochs('./epochs_tensors_connectivity', connectivity=True,
                           connectivity_freqs=[4, 8, 12, 18, 30],
                           connectivity_method='pli',
                           baseline = None, tmin = 0, tmax = 4.1)