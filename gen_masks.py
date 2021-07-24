import os
from tqdm import tqdm
import numpy as np
from scipy import ndimage

import torch
from torch.utils.data import DataLoader

from data.dataset import ISPRS_Dataset
from models.Unet import UNET

configs = {
    'path_to_weakly_masks': 'data/preprocessed/weakly_masks',
    'path_to_weak_mask_erosion': 'data/preprocessed/weakly_masks_erosion',
    'model_path': 'checkpoints/baseline_Unet.pth'
}


def makedirs(dirs: str):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def erosion(img):
    img = img.copy()
    for i in range(4):
        img[i] = ndimage.binary_dilation(img[i], iterations=10).astype(int)

    orig_img = img.copy()
    for i in range(4):
        img[i] = ndimage.binary_erosion(img[i], iterations=20).astype(int)

    return img - (orig_img - img)


def get_masks(model, dataloader, device, path_to_weakly_masks, path_to_weak_mask_erosion):
    makedirs(path_to_weakly_masks)
    makedirs(path_to_weak_mask_erosion)

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

            np.save(f'{path_to_weakly_masks}/{file_names[i]}', pred_masks[i])

            pred_masks[i] = erosion(pred_masks[i])
            np.save(f'{path_to_weak_mask_erosion}/{file_names[i]}', pred_masks[i])


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNET(in_channels=3, out_channels=5)
    model.load_state_dict(torch.load(configs['model_path']))
    model.to(device)

    dataset = ISPRS_Dataset('data/preprocessed', 'data/preprocessed/metadata.csv', 'weak_train')
    dataloader = DataLoader(dataset, shuffle=False, batch_size=3)

    get_masks(
        model=model,
        dataloader=dataloader,
        device=device,
        path_to_weakly_masks=configs['path_to_weakly_masks'],
        path_to_weak_mask_erosion=configs['path_to_weak_mask_erosion']
    )
