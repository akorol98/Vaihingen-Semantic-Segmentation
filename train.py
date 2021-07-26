from comet_ml import Experiment

import numpy as np
from tqdm import tqdm
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ISPRS_Dataset
from models.Unet import UNET
from tools.metrics import IoU

# Cometml experiment initialization
experiment = Experiment(
    api_key='icIskpBA5S6z2xa1l7Fxa1uX2',
    project_name="Vaihingen-Semantic-Segmentation",
    workspace="akorol",
)

configs = {
    'batch_size': 4,
    'lr': 0.0001,
    'n_epochs': 100,
    'num_workers': 4,
    'weight_decay': 1e-8,
    'seed': 42,
    'split': 'train',
    'train_mode': 'train',
    'path_to_save': 'checkpoints/',
    'path_train_output': 'train_out/',
    'model_name': 'baseline_augment_Unet.pth',
    'path_to_checkpoint': 'checkpoints/pretrained_Unet.pth'
}

# set malual seed for reproductivity
torch.manual_seed(configs['seed'])
np.random.seed(configs['seed'])

if not os.path.exists(configs['path_to_save']):
    os.makedirs(configs['path_to_save'])

if not os.path.exists(configs['path_train_output']):
    os.makedirs(configs['path_train_output'])


def train(model, device, epochs, bs, lr, wd, nw, split, train_mode):

    dataset = ISPRS_Dataset('data/preprocessed', 'data/preprocessed/metadata.csv',
                            split=split, train_mode=train_mode)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=bs, num_workers=nw)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()

    step = 0
    losses = []
    ious = []
    for i, epoch in enumerate(range(epochs)):
        model.train()

        epoch_loss = 0
        for batch in tqdm(dataloader, desc='Training'):
            imgs = batch['img']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            if configs['train_mode'] == 'weakly_train_erosion':
                pred_masks = model(imgs, true_masks)
                true_masks[true_masks <= -1] = 0
            else:
                pred_masks = model(imgs)

            loss = criterion(pred_masks, true_masks)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            experiment.log_metric("Loss", loss.item(), step=step)
            step += 1

        validation_iou = validation(model, device)
        train_iou = validation(model, device, subset='train')

        print(f'Epoch: {i + 1}, loss: {epoch_loss / len(dataloader)}, '
              f'mIoU_train: {train_iou}, mIoU_validation: {validation_iou}')

        experiment.log_metric("mIoU_train", train_iou, step=step)
        experiment.log_metric("mIoU_validation", validation_iou, step=step)

        losses.append(epoch_loss / len(dataloader))
        ious.append(validation_iou)

    return np.array(losses), np.array(ious)


def validation(model, device, subset='validation'):
    model.eval()

    dataset = ISPRS_Dataset('data/preprocessed', 'data/preprocessed/metadata.csv', subset)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    ious = []
    for batch in tqdm(dataloader, desc='IoU ' + subset):
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

    if sys.argv[1] == 'baseline':
        configs['path_to_checkpoint'] = None
        configs['model_name'] = 'baseline_Unet.pth'
        configs['n_epochs'] = 20
        configs['split'] = 'train'
        configs['train_mode'] = 'train'
    elif sys.argv[1] == 'supervised':
        configs['path_to_checkpoint'] = None
        configs['model_name'] = 'baseline_supervised_Unet.pth'
        configs['n_epochs'] = 100
        configs['split'] = 'weak_train'
        configs['train_mode'] = 'baseline_supervised'
    elif sys.argv[1] == 'weakly_supervised':
        configs['path_to_checkpoint'] = None
        configs['model_name'] = 'final_Unet.pth'
        configs['n_epochs'] = 100
        configs['split'] = 'train'
        configs['train_mode'] = 'train_with_weakly'
    else:
        assert False, 'Wrong argument! One of the following values is available: ' \
                      '["baseline", "pretrain", "finetune"]'

    if len(sys.argv) == 3:
        configs['n_epochs'] = int(sys.argv[2])

    experiment.log_parameters(configs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNET(in_channels=3, out_channels=5)
    if configs['path_to_checkpoint']:
        model.load_state_dict(torch.load(configs['path_to_checkpoint']))
    model.to(device)

    losses, ious = train(
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

    torch.save(model.state_dict(), configs['path_to_save'] + configs['model_name'])

    # np.save(configs['path_train_output']+'losses_'+sys.argv[1]+'.npy', losses)
    # np.save(configs['path_train_output']+'ious_'+sys.argv[1]+'.npy', ious)
