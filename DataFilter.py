


from pathlib import Path
import os
import shutil
import torch

class DataFilter:

    def __init__(self):
        pass


    def filter(self, data_path: Path | str, save_path: Path | str,
               data_type: str, intrasubject: bool, subject_list: list[int] | None,
               filter_0_class: bool):

        if isinstance(data_path, str):
            data_path = Path(data_path)

        if isinstance(save_path, str):
            save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        labels = []
        for subject_id in subject_list:
            if intrasubject:
                new_save_path = save_path / str(subject_id).zfill(3)
                new_save_path.mkdir(parents=True, exist_ok=True)
                new_labels = self._filter(data_path, new_save_path, data_type, subject_id, filter_0_class)
                labels_tensor = torch.tensor(new_labels)
                torch.save(labels_tensor, new_save_path / 'labels.pt')
            else:
                new_labels = self._filter(data_path, save_path, data_type, subject_id, filter_0_class)
                labels.extend(new_labels)

        if not intrasubject:
            labels_tensor = torch.tensor(labels)
            torch.save(labels_tensor, save_path / 'labels.pt')



    def _filter(self, data_path: Path | str, save_path: Path | str,
               data_type: str, subject_id: int, filter_0_class: bool):

        if data_type == 'real':
            filter_type = ['r']
        elif data_type == 'imagery':
            filter_type = ['i']
        else:
            filter_type = ['i', 'r']


        labels = []
        file_list = os.listdir(data_path)
        for i, file_name in enumerate(file_list):

            label = int(file_name[-4])

            if filter_0_class and (label == 0):
                continue

            if (file_name[0] in filter_type) and \
                    (int(file_name[2:5]) == subject_id):

                shutil.copy2(data_path / file_name, save_path)
                labels.append(label-1)

        return labels

if __name__ == '__main__':
    data_filter = DataFilter()
    """
    data_filter.filter('./epochs_tensors/', './working_data_inter',
                       data_type='both', intrasubject=False, subject_list=list(range(1, 110)),
                       filter_0_class=True)
    """

    data_filter.filter('./epochs_tensors_connectivity/', './working_data_connectivity',
                       data_type='both', intrasubject=True, subject_list=list(range(1, 110)),
                       filter_0_class=True)





