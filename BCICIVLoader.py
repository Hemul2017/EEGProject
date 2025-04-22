from pathlib import Path
import mne
import os
import re


def BCICIVLoader(load_path: Path | str, pattern: str):
    if isinstance(load_path, str):
        load_path = Path(load_path)

    for participant_dir in os.listdir(load_path):
        if re.match(pattern, participant_dir):
            participant_dir = load_path / participant_dir
            for file_name in os.listdir(participant_dir):
                if file_name.endswith('.edf'):
                    load_file_path = participant_dir / file_name
                    raw_data = mne.io.read_raw_edf(load_file_path)
                    epochs = mne.Epochs(raw_data, baseline=None, tmin=0, tmax=4.1)
                    yield epochs, file_name




if __name__ == '__main__':
    mne.io._read
    with open('./BCICIV_data/B0101T.gdf', mode='rb') as file:
        print(file.read(80).decode("latin-1").rstrip())