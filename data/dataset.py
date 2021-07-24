import ast
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ISPRS_Dataset(Dataset):

    def __init__(self, data_path, metadata_path, split='train', train_mode='train'):
        self.data_path = data_path
        self.train_mode = train_mode

        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[
            self.metadata['split'] == split
            ].reset_index(drop=True)

        self.metadata.label = self.metadata.label.map(ast.literal_eval)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata['img'][idx]
        tile = self.metadata['tile'][idx]

        img = np.load(f'{self.data_path}/imgs/{img_name + tile}.npy')
        if self.train_mode == 'train':
            mask = np.load(f'{self.data_path}/masks/{img_name + tile}.npy')
        elif self.train_mode == 'weakly_train':
            mask = np.load(f'{self.data_path}/weakly_masks/{img_name + tile}.npy')
        elif self.train_mode == 'weakly_train_erosion':
            mask = np.load(f'{self.data_path}/weakly_masks_erosion/{img_name + tile}.npy')
        else:
            assert False, 'Wrong training mode! One of the following values is available: ' \
                          '["train", "weakly_train", "weakly_train_erosion"]'
        label = np.array(self.metadata['label'][idx])

        img = img / 255

        return {
            'img': img,
            'mask': mask,
            'label': label,
            'file_name': img_name + tile + '.npy'
        }
