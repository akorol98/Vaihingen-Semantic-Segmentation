import numpy as np
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ISPRS_Dataset
from models.Unet import UNET
from tools.metrics import IoU

configs = {
    'batch_size': 3,
    'lr': 0.0001,
    'n_epochs': 1,
    'num_workers': 0,
    'weight_decay': 1e-8,
    'split': 'train',
    'train_mode': 'train',
    'path_to_save': 'checkpoints/'
}

if not os.path.exists(configs['path_to_save']):
    os.makedirs(configs['path_to_save'])


def train(model, device, epochs, bs, lr, wd, nw, split, train_mode):
    dataset = ISPRS_Dataset('data/preprocessed', 'data/preprocessed/metadata.csv',
                            split=split, train_mode=train_mode)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=bs, num_workers=nw)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()

    for i, epoch in enumerate(range(epochs)):
        model.train()

        epoch_loss = 0
        for batch in tqdm(dataloader, desc='Training'):
            imgs = batch['img']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            pred_masks = model(imgs)
            loss = criterion(pred_masks, true_masks)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

        validation_iou = validation(model, device)
        train_iou = validation(model, device, subset='train')
        print(f'Epoch: {i + 1}, loss: {epoch_loss / len(dataloader)}, '
              f'IoU_train: {train_iou}, IoU_validation: {validation_iou}')


def validation(model, device, subset='validation'):
    model.eval()

    dataset = ISPRS_Dataset('data/preprocessed', 'data/preprocessed/metadata.csv', subset)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    ious = []
    for batch in tqdm(dataloader, desc='Validation'):
        imgs = batch['img']
        imgs = imgs.to(device=device, dtype=torch.float32)
        mask = batch['mask'].numpy()

        with torch.no_grad():
            pred_masks = model(imgs).cpu().detach().numpy()
        pred_masks[pred_masks > 0] = 1
        pred_masks[pred_masks < 0] = 0

        ious.append(IoU(pred_masks, mask))

    return np.mean(ious)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNET(in_channels=3, out_channels=5)
    model.to(device)

    train(
        model=model,
        device=device,
        epochs=configs['n_epochs'],
        bs=configs['batch_size'],
        lr=configs['lr'],
        wd=configs['weight_decay'],
        nw=configs['num_workers'],
        split=configs['split'],
        train_mode=configs['train_mode']
    )

    torch.save(model.state_dict(), configs['path_to_save'] + 'baseline_Unet.pth')
