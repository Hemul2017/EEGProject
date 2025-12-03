from pathlib import Path
import mne
import os
import re
import torch
import numpy as np



def EEGDataLoader(load_path: Path | str, pattern: str,
                  calc_adj_matrix: bool, **kwargs):
    if isinstance(load_path, str):
        load_path = Path(load_path)

    for participant_dir in os.listdir(load_path):

        if re.match(pattern, participant_dir):
            participant_dir = load_path / participant_dir

            for file_name in os.listdir(participant_dir):

                if file_name.endswith('.edf'):
                    load_file_path = participant_dir / file_name
                    raw_data = mne.io.read_raw_edf(load_file_path)

                    if calc_adj_matrix:
                        get_adj_matrix(raw_data)
                        calc_adj_matrix = False

                    epochs = mne.Epochs(raw_data, **kwargs)
                    yield epochs, file_name


def get_adj_matrix(raw_data: mne.io.Raw) -> None:
    raw_data = encode_channels(raw_data)
    raw_data.set_montage('biosemi64')
    adj_matrix, ch_names = mne.channels.find_ch_adjacency(raw_data.info, ch_type='eeg')
    adj_matrix = torch.tensor(adj_matrix.toarray())
    ch_names = np.array(ch_names)
    print(adj_matrix)
    print(ch_names)
    save_path = Path('./info')
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(adj_matrix, save_path / 'physionet_adj_matrix.pt')
    np.save(save_path / 'physionet_ch_names.npy', ch_names)



# Кодирует каналы в соответствии со схемой 10-10 (biosemi64 согласно mne)
def encode_channels(raw_data: mne.io.Raw) -> mne.io.Raw:
    ch_mapping = {}
    for ch_name in raw_data.ch_names:
        new_ch_name = ch_name.rstrip('.').upper()
        if new_ch_name.endswith('Z'):
            new_ch_name = new_ch_name[:-1] + new_ch_name[-1].lower()
        if new_ch_name == 'T10':
            new_ch_name = 'P10'
        elif new_ch_name == 'T9':
            new_ch_name = 'P9'
        elif new_ch_name == 'FPz':
            new_ch_name = 'Fpz'
        elif new_ch_name == 'FP2':
            new_ch_name = 'Fp2'
        elif new_ch_name == 'FP1':
            new_ch_name = 'Fp1'
        ch_mapping |= {ch_name: new_ch_name}
    raw_data.rename_channels(ch_mapping)

    return raw_data



if __name__ == '__main__':
    generator = EEGDataLoader(load_path='./files', pattern=r'\w\d{3}', calc_adj_matrix=True,
                              baseline=None, tmin=0, tmax=4.1)
    print(generator)
    for i in range(5):
        epoch = next(generator)[0]
        print(epoch.get_data().shape)
        print(epoch.get_annotations_per_epoch())









