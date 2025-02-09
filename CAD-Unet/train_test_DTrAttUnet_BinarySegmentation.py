# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as TF
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

##########################################   
##########################################  
##########################################
from torch.utils.data import Dataset
import os
import os.path
import numpy as np
import torch
import cv2


class Data_loaderVE(Dataset):
    def __init__(self, root, train, transform=None):

        self.train = train
        self.data, self.y, self.y2 = torch.load(os.path.join(root, train))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img1, y1, y2 = self.data[index], self.y[index], self.y2[index]
        img1 = np.array(img1)
        y1 = np.array(y1)
        y2 = np.array(y2)
        img1 = img1.astype(np.uint8)
        y1 = y1.astype(np.uint8)
        y2 = y2.astype(np.uint8)

        #In binary classification, all values greater than 0 in the labels are set to 1.
        y1[y1 > 0.0] = 1.0
        y2[y2 > 0.0] = 1.0

        # Process the labels of infection edges.
        y33 = y2 * 255.0
        y33 = y33.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        y3 = cv2.morphologyEx(y33, cv2.MORPH_GRADIENT, kernel)
        y3[y3 > 0.0] = 1.0

        # Data augmentation
        if self.transform is not None:
            augmentations = self.transform(image=img1, masks=[y1, y2, y3])
            image = augmentations["image"]
            mask = augmentations["masks"][0]
            mask1 = augmentations["masks"][1]
            edge = augmentations["masks"][2]

        return image, mask1, mask, edge

    def __len__(self):
        return len(self.data)

    ############################
img_size = 224
train_transform = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.Normalize(  # 标准化
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)
val_transforms = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)
root_path=''
train_dataset_name=''
valida_dataset_name=''
############################
# Create datasets.
train_set = Data_loaderVE(
    root=root_path
    , train=train_dataset_name
    , transform=train_transform
)

validate_set = Data_loaderVE(
    root=root_path
    , train=valida_dataset_name
    , transform=val_transforms
)
############################
F1_mean, dise_mean, IoU_mean = [], [], []
Recall_mean = []

dataset_idx = "data4"

modl_n = 'CADUnet'

epochs = 120
batch_size = 6
num_hands = 16
rate = 0.0

runs = 5
for itr in range(runs):
    # itr += 5
    model_sp = "./Bin_Models" + str(dataset_idx) +  "/" + modl_n + "/Models"
    if not os.path.exists(model_sp):
        os.makedirs(model_sp)
    ############################

    name_model_final = model_sp + '/' + str(itr) + '_fi.pt'
    name_model_bestF1 = model_sp + '/' + str(itr) + '_bt.pt'

    model_spR = "./Bin_Models" + str(dataset_idx) + "/" + modl_n + "/Results"
    if not os.path.exists(model_spR):
        os.makedirs(model_spR)

    training_tsx = model_spR + '/' + str(itr) + '.txt'

    criterion = nn.BCEWithLogitsLoss()

    #######create model
    from models import Architecture_CADUnet as networks
    model = getattr(networks, modl_n)(in_channels=3, out_channels=1, img_size=224)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    #######

    torch.set_grad_enabled(True)

    ############################
    # create dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda:0")

    start = time.time()
    model.to(device)

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    train_dise, valid_dise = [], []
    train_dise2, valid_dise2 = [], []

    train_IoU, valid_IoU = [], []

    train_F1score, valid_F1score = [], []

    train_Spec, valid_Spec = [], []
    train_Sens, valid_Sens = [], []
    train_Prec, valid_Prec = [], []

    epoch_count = []

    best_F1score = -1

    segm = nn.Sigmoid()

    for epoch in range(epochs):
        epoch_count.append(epoch)
        lrr=1e-4
        if epoch >50:
            lrr=1e-5
        if epoch > 90:
            lrr = 1e-6

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
                dataloader = train_loader
            else:
                model.train(False)
                dataloader = validate_loader

            running_loss = 0.0

            num_correct = 0
            num_pixels = 0

            step = 0
            optimizer = torch.optim.Adam(model.parameters(), lr=lrr)

            dice_scores = 0
            dice_scores2 = 0
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for batch in tqdm(dataloader):
                x, y, y2, edge = batch
                x = x.to(device)
                y = y.float().to(device)
                y2 = y2.float().to(device)
                edge = edge.float().to(device)
                unique_values1 = torch.unique(y)
                unique_values2 = torch.unique(y2)
                unique_values3 = torch.unique(edge)
                step += 1
                if phase == 'train':
                    # Forward propagation
                    outputs,outputs2= model(x)
                    # Calculate the hybrid loss.
                    loss1 = criterion(outputs.squeeze(dim=1), y)
                    loss2 = criterion(outputs2.squeeze(dim=1), y2)
                    loss3 = criterion(outputs.squeeze(dim=1)*edge, edge)
                    loss = 0.7*loss1 + 0.3*loss2+ 1*loss3
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs,outputs2= model(x)
                        loss1 = criterion(outputs.squeeze(dim=1), y)

                running_loss += loss1

                preds = segm(outputs) > 0.5
                preds = preds.squeeze(dim=1).cpu().numpy().astype(int)
                yy = y > 0.5
                yy = yy.squeeze(dim=1).cpu().numpy().astype(int)
                num_correct += np.sum(preds == yy)
                # Calculate TP，TN，FP，FN
                TP += np.sum(((preds == 1).astype(int) +
                              (yy == 1).astype(int)) == 2)
                TN += np.sum(((preds == 0).astype(int) +
                              (yy == 0).astype(int)) == 2)
                FP += np.sum(((preds == 1).astype(int) +
                              (yy == 0).astype(int)) == 2)
                FN += np.sum(((preds == 0).astype(int) +
                              (yy == 1).astype(int)) == 2)
                num_pixels += preds.size
                for idice in range(preds.shape[0]):
                    dice_scores += (2 * (preds[idice] * yy[idice]).sum()) / (
                            (preds[idice] + yy[idice]).sum() + 1e-8
                    )

                predss = np.logical_not(preds).astype(int)
                yyy = np.logical_not(yy).astype(int)
                for idice in range(preds.shape[0]):
                    dice_sc1 = (2 * (preds[idice] * yy[idice]).sum()) / (
                            (preds[idice] + yy[idice]).sum() + 1e-8
                    )
                    dice_sc2 = (2 * (predss[idice] * yyy[idice]).sum()) / (
                            (predss[idice] + yyy[idice]).sum() + 1e-8
                    )
                    dice_scores2 += (dice_sc1 + dice_sc2) / 2

                del x
                del y

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc2 = (num_correct / num_pixels) * 100
            epoch_dise = dice_scores / len(dataloader.dataset)
            epoch_dise2 = dice_scores2 / len(dataloader.dataset)
            # 计算一堆指标
            Spec = 1 - (FP / (FP + TN))
            Sens = TP / (TP + FN)
            Prec = TP / (TP + FP + 1e-8)
            F1score = TP / (TP + ((1 / 2) * (FP + FN)) + 1e-8)
            IoU = TP / (TP + FP + FN)  # 计算IOU

            if phase == 'valid':
                if F1score > best_F1score:
                    best_F1score = F1score
                    torch.save(model.state_dict(), name_model_bestF1)

            with open(training_tsx, "a") as f:

                print('Epoch {}/{}'.format(epoch, epochs - 1), file=f)
                print('-' * 10, file=f)
                print(
                    '{} Loss: {:.4f} Acc: {:.8f} Dise: {:.8f} Dise2: {:.8f} IoU: {:.8f} F1: {:.8f} Spec: {:.8f} Sens: {:.8f} Prec: {:.8f}'
                    .format(phase, epoch_loss, epoch_acc2, epoch_dise, epoch_dise2, IoU, F1score, Spec, Sens, Prec),
                    file=f)
            # 保存一堆训练结果
            train_loss.append(np.array(epoch_loss.detach().cpu())) if phase == 'train' \
                else valid_loss.append(np.array(epoch_loss.detach().cpu()))
            train_acc.append(np.array(epoch_acc2)) if phase == 'train' \
                else valid_acc.append((np.array(epoch_acc2)))
            train_dise.append(np.array(epoch_dise)) if phase == 'train' \
                else valid_dise.append((np.array(epoch_dise)))
            train_dise2.append(np.array(epoch_dise2)) if phase == 'train' \
                else valid_dise2.append((np.array(epoch_dise2)))

            train_IoU.append(np.array(IoU)) if phase == 'train' \
                else valid_IoU.append((np.array(IoU)))

            train_F1score.append(np.array(F1score)) if phase == 'train' \
                else valid_F1score.append((np.array(F1score)))

            train_Spec.append(np.array(Spec)) if phase == 'train' \
                else valid_Spec.append((np.array(Spec)))
            train_Sens.append(np.array(Sens)) if phase == 'train' \
                else valid_Sens.append((np.array(Sens)))
            train_Prec.append(np.array(Prec)) if phase == 'train' \
                else valid_Prec.append((np.array(Prec)))

    torch.save(model.state_dict(), name_model_final)
    time_elapsed = time.time() - start

    with open(training_tsx, "a") as f:
        # print(model, file=f)
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), file=f)

    ############################
    with open(training_tsx, "a") as f:

        print('Train', file=f)
        print('Train F1 score', file=f)
        print(train_F1score[valid_F1score.index(np.max(valid_F1score))], file=f)

        print(train_acc[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_dise[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_dise2[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_IoU[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_Sens[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_Spec[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_Prec[valid_F1score.index(np.max(valid_F1score))], file=f)

        print('-' * 10, file=f)
        print('train Results', file=f)
        print('train_acc', train_acc, file=f)
        print('train_F1score', train_F1score, file=f)
        print('train_IoU', train_IoU, file=f)
        print('train_Sens', train_Sens, file=f)
        print('train_Spec', train_Spec, file=f)
        print('train_Prec', train_Prec, file=f)

        print('-' * 10, file=f)
        print('Train', file=f)
        print('Best Val F1 score', file=f)
        print(np.max(valid_F1score), file=f)
        print('Index of Best', file=f)
        print(name_model_bestF1, file=f)
        print(valid_F1score.index(np.max(valid_F1score)), file=f)

        print('-' * 10, file=f)
        print('Val Results', file=f)
        print('valid_acc', valid_acc[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_F1score', valid_F1score[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_IoU', valid_IoU[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_Sens', valid_Sens[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_Spec', valid_Spec[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_Prec', valid_Prec[valid_F1score.index(np.max(valid_F1score))], file=f)

        print('-' * 10, file=f)
        print('Val Results', file=f)
        print('valid_acc', valid_acc, file=f)
        print('valid_F1score', valid_F1score, file=f)
        print('valid_IoU', valid_IoU, file=f)
        print('valid_Sens', valid_Sens, file=f)
        print('valid_Spec', valid_Spec, file=f)
        print('valid_Prec', valid_Prec, file=f)
        F1_mean.append(valid_F1score[valid_F1score.index(np.max(valid_F1score))])
        IoU_mean.append(valid_IoU[valid_F1score.index(np.max(valid_F1score))])
        Recall_mean.append(valid_Sens[valid_F1score.index(np.max(valid_F1score))])

    f.close()

std1 = np.std(F1_mean)
std2 = np.std(dise_mean)
std3 = np.std(IoU_mean)
std4 = np.std(Recall_mean)

training_tsx = model_spR + '/' + 'mean' + '.txt'
F1_mean.append(np.mean(F1_mean))
IoU_mean.append(np.mean(IoU_mean))
Recall_mean.append(np.mean(Recall_mean))

F1_mean.append(std1)
IoU_mean.append(std3)
Recall_mean.append(std4)
with open(training_tsx, "a") as f:
    print('F1_mean', F1_mean, file=f)
    print('IoU_mean', IoU_mean, file=f)
    print('Recall_mean', Recall_mean, file=f)

f.close()
