


from typing import Iterator
import mne
import pandas as pd
from tqdm import tqdm
from DataLoader import EEGDataLoader
from pathlib import Path
import time
import torch

class EEGDataSaver:

    def __init__(self, load_path: Path | str, pattern: str):
        self.load_path = load_path
        self.pattern = pattern
        self.first_type_runs = {'03', '04', '07', '08', '11', '12'}
        self.encode_dict = {'rest': 0, 'left_hand': 1, 'right_hand': 2, 'both_hands': 3, 'feet': 4}

    def save_to_2d_dataframe_by_epochs(self, save_path: Path | str, distinguish_imagined: bool = True) -> None:

        if isinstance(save_path, str):
            save_path = Path(save_path)

        data_loader = EEGDataLoader(self.load_path, self.pattern)

        labels = []

        for file_idx, (epochs, file_name) in enumerate(data_loader):
            n_run = file_name.split('.')[0][-2:]
            # Пропуск базовых сессий
            if (n_run == '01') or (n_run == '02'):
                continue
            # Разделение данных на представляемые и реальные движения
            if distinguish_imagined:
                if int(n_run) % 2 == 0:
                    save_path = save_path / 'imaginary'
                else:
                    save_path = save_path / 'real'

            epoch_labels = []
            for epoch_idx, (evoked, epoch_label) in enumerate(zip(epochs.iter_evoked(), epochs.get_annotations_per_epoch())):
                df = evoked.to_data_frame(scalings={'eeg': 1})
                if not df.shape[0] == 657:
                    print(f'Wrong DataFrame size {df.shape[0]}')
                    continue
                df.drop('time', axis=1, inplace=True)
                save_path.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path / f'{file_idx:04}_{epoch_idx:02}.csv')
                epoch_label = self._encode_labels(epoch_label[0][2], n_run)
                epoch_labels.append(epoch_label)

            labels.extend(epoch_labels)
            print(f'\nFile {file_name} has successfully been processed\n')

        labels_df = pd.Series(labels)
        labels_df.to_csv(save_path / 'labels.csv')


    def save_to_2d_tensors_by_epochs(self, base_save_path: Path | str) -> None:

        if isinstance(base_save_path, str):
            base_save_path = Path(base_save_path)

        data_loader = EEGDataLoader(self.load_path, self.pattern)

        labels_imaginary = []
        labels_real = []
        n_samples_expected = 657

        for file_idx, (epochs, file_name) in enumerate(data_loader):
            save_path = base_save_path
            n_run = file_name.split('.')[0][-2:]
            # Пропуск базовых сессий
            if (n_run == '01') or (n_run == '02'):
                continue
            # Разделение данных на представляемые и реальные движения
            if int(n_run) % 2 == 0:
                save_path = save_path / 'imaginary'
                labels = labels_imaginary
            else:
                save_path = save_path / 'real'
                labels = labels_real

            epoch_labels = []
            for epoch_idx, (evoked, epoch_label) in enumerate(zip(epochs.iter_evoked(), epochs.get_annotations_per_epoch())):
                data = evoked.get_data()
                if data.shape[1] != n_samples_expected:
                    print(f'Record {file_idx:04}_{epoch_idx:02} has {data.shape[1]} samples, while {n_samples_expected} samples expected')
                    continue
                data_tensor = torch.tensor(data, dtype=torch.float32)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(data_tensor, save_path / f'{file_idx:04}_{n_run}_{epoch_idx:02}.pt')
                epoch_label = self._encode_labels(epoch_label[0][2], n_run)
                epoch_labels.append(epoch_label)

            labels.extend(epoch_labels)
            print(f'\nFile {file_name} has successfully been processed\n')

        imaginary_labels_tensor = torch.tensor(labels_imaginary)
        torch.save(imaginary_labels_tensor, base_save_path / 'imaginary' / 'labels.pt')
        real_labels_tensor = torch.tensor(labels_real)
        torch.save(real_labels_tensor, base_save_path / 'real' / 'labels.pt')

        print(f'\nAll files have successfully been processed\n')

    # Тензоры с разделением на воображаемое и реальное движение
    def save_to_2d_tensors_by_epochs(self, base_save_path: Path | str) -> None:

        if isinstance(base_save_path, str):
            base_save_path = Path(base_save_path)

        data_loader = EEGDataLoader(self.load_path, self.pattern)

        labels_imaginary = []
        labels_real = []
        n_samples_expected = 657

        for file_idx, (epochs, file_name) in enumerate(data_loader):
            save_path = base_save_path
            n_run = file_name.split('.')[0][-2:]
            # Пропуск базовых сессий
            if (n_run == '01') or (n_run == '02'):
                continue
            # Разделение данных на представляемые и реальные движения
            if int(n_run) % 2 == 0:
                save_path = save_path / 'imaginary'
                labels = labels_imaginary
            else:
                save_path = save_path / 'real'
                labels = labels_real

            epoch_labels = []
            for epoch_idx, (evoked, epoch_label) in enumerate(zip(epochs.iter_evoked(), epochs.get_annotations_per_epoch())):
                data = evoked.get_data()
                if data.shape[1] != n_samples_expected:
                    print(f'Record {file_idx:04}_{epoch_idx:02} has {data.shape[1]} samples, while {n_samples_expected} samples expected')
                    continue
                data_tensor = torch.tensor(data, dtype=torch.float32)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(data_tensor, save_path / f'{file_idx:04}_{n_run}_{epoch_idx:02}.pt')
                epoch_label = self._encode_labels(epoch_label[0][2], n_run)
                epoch_labels.append(epoch_label)

            labels.extend(epoch_labels)
            print(f'\nFile {file_name} has successfully been processed\n')

        imaginary_labels_tensor = torch.tensor(labels_imaginary)
        torch.save(imaginary_labels_tensor, base_save_path / 'imaginary' / 'labels.pt')
        real_labels_tensor = torch.tensor(labels_real)
        torch.save(real_labels_tensor, base_save_path / 'real' / 'labels.pt')

        print(f'\nAll files have successfully been processed\n')

    # Тензоры без разделения на воображаемое и реальное движение
    def save_to_2d_tensors_by_epochs_wo_division(self, save_path: Path | str) -> None:

        if isinstance(save_path, str):
            save_path = Path(save_path)

        data_loader = EEGDataLoader(self.load_path, self.pattern)

        labels = []
        n_samples_expected = 657

        for file_idx, (epochs, file_name) in enumerate(data_loader):
            n_run = file_name.split('.')[0][-2:]
            if (n_run == '01') or (n_run == '02'):
                continue
            epoch_labels = []
            for epoch_idx, (evoked, epoch_label) in enumerate(zip(epochs.iter_evoked(), epochs.get_annotations_per_epoch())):
                data = evoked.get_data()
                if data.shape[1] != n_samples_expected:
                    print(f'Record {file_idx:04}_{epoch_idx:02} has {data.shape[1]} samples, while {n_samples_expected} samples expected')
                    continue
                data_tensor = torch.tensor(data, dtype=torch.float32)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(data_tensor, save_path / f'{file_idx:04}_{epoch_idx:02}.pt')
                epoch_label = self._encode_labels(epoch_label[0][2], n_run)
                epoch_labels.append(epoch_label)

            labels.extend(epoch_labels)
            print(f'\nFile {file_name} has successfully been processed\n')

        labels_tensor = torch.tensor(labels)
        torch.save(labels_tensor, save_path / 'labels.pt')

        print(f'All files has successfully been processed')


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
    #data_saver = EEGDataSaver(load_path='./files', pattern=r'\w\d{3}')
    #data_saver.save_to_2d_tensors_by_epochs('./epochs_tensors')
    data_saver = EEGDataSaver(load_path='./files', pattern=r'\w\d{3}')
    data_saver.save_to_2d_tensors_by_epochs_wo_division('./epochs_tensors_wo_division')


