import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data.dataset import ISPRS_Dataset
from models.Unet import UNET

configs = {
    'path_to_save': 'data/weakly_masks',
    'model_path': 'checkpoints/baseline_Unet.pth'
}


def get_masks(model, dataloader, device, path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    model.eval()

    for batch in tqdm(dataloader, desc='Gen masks'):

        file_names = batch['file_name']

        imgs = batch['img']
        imgs = imgs.to(device=device, dtype=torch.float32)

        img_label = batch['label']

        with torch.no_grad():
            pred_masks = model(imgs).cpu().detach().numpy()
        pred_masks[pred_masks > 0] = 1
        pred_masks[pred_masks < 0] = 0

        # correct mask with image label
        pred_masks[img_label == 0] = 0

        for i in range(0, len(imgs)):
            np.save(f'{path_to_save}/{file_names[i]}', pred_masks[i])


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNET(in_channels=3, out_channels=5)
    model.load_state_dict(torch.load(configs['model_path']))
    model.to(device)

    dataset = ISPRS_Dataset('data/preprocessed', 'data/preprocessed/metadata.csv', 'weak_train')
    dataloader = DataLoader(dataset, shuffle=False, batch_size=3)

    get_masks(model, dataloader, device, configs['path_to_save'])
