# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

img_size = 224
batch_size = 6
epochs = 120
rate=0.0
num_hands=16
##########################################   

from torch.utils.data import Dataset
import os.path


##########################################   
class Data_loaderV(Dataset):
    def __init__(self, root, train, transform=None):

        self.train = train  # training set or test set
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


        y1[y1 > 0.0] = 1.0 # Lung
        y2[y2 > 1.0] = 2.0 # Multi Class
        # Data augmentation
        if self.transform is not None:
            augmentations = self.transform(image=img1, mask=y1, mask0 = y2)
            image = augmentations["image"]
            mask = augmentations["mask"]
            mask1 = augmentations["mask0"]

            
        return   image, mask1, mask

    def __len__(self):
        return len(self.data)   
    
########################################## 

train_transform = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],additional_targets={'mask0': 'mask'}
)
######
val_transforms = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],additional_targets={'mask0': 'mask'}
)

############################
root_path=''
train_dataset_name=''
valida_dataset_name=''
############################
# Create datasets.
train_set = Data_loaderV(
    root=root_path
    , train=train_dataset_name
    , transform=train_transform
)

validate_set = Data_loaderV(
    root=root_path
    , train=valida_dataset_name
    , transform=val_transforms
)

############################
# Part 4
# 
F1_mean1, dise_mean1, IoU_mean1 = [], [], []
F1_mean2, dise_mean2, IoU_mean2 = [], [], []
Recall_mean1,Recall_mean2=[],[]
Spec_mean1,Spec_mean2 = [], []
Prec_mean1,Prec_mean2=[],[]
acc_mean=[]

############################
dataset_idx ="MulitSeg"
runs = 5
modl_n = 'CADUnet'



for itr in range(runs):
    
    model_sp = "./" + dataset_idx +"/" + modl_n + "/Models"
    if not os.path.exists(model_sp):
        os.makedirs(model_sp)
    ############################
    
    name_model_final = model_sp+ '/' + str(itr) + '_fi.pt'
    name_model_bestF1 =  model_sp+ '/' + str(itr) + '_bt.pt'
    
    model_spR = "./" + dataset_idx+"/" + modl_n+ "/Results"
    if not os.path.exists(model_spR):
        os.makedirs(model_spR)
        
    training_tsx = model_spR+ '/' + str(itr) + '.txt'
       
    criterion =nn.BCEWithLogitsLoss()
    criterion1 = nn.CrossEntropyLoss()
    reconstruction_loss = nn.MSELoss(reduction="none")
    #######
    from models import Architecture_CADUnet as networks
    model = getattr(networks, modl_n)(in_channels=3, out_channels=3, img_size=224,num_heads=num_hands,att_ratio=rate)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    #######
    
    torch.set_grad_enabled(True)

    #####
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda:0")
    
    start = time.time()
    model.to(device)
    
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    train_dise1,train_dise2, valid_dise1, valid_dise2 = [], [], [], []
    train_IoU1, train_IoU2, valid_IoU1, valid_IoU2  = [], [], [], []
    valid_Recall1,valid_Recall2 = [], []
    valid_Spec1,valid_Spec2 = [], []
    valid_Prec1,valid_Prec2=[],[]
    train_F1score1, train_F1score2, valid_F1score1, valid_F1score2 = [], [], [], []    
    
    epoch_count = []
    
    best_F1score = -1
    
    segm = nn.Sigmoid()
    
    
    for epoch in range(epochs):
        epoch_count.append(epoch)

    
        lr = 0.01
        if epoch > 60:
            lr = 0.001
        if epoch > 90:
            lr = 0.0001
    
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
                dataloader = train_loader
            else:
                model.train(False)
                dataloader = validate_loader
    
            running_loss = 0.0
            
            num_correct1 = 0
            num_correct2 = 0
            num_correct3 = 0
            num_correct4 = 0
            num_pixels = 0            
    
            step = 0
    

            dice_scores_1 = 0
            dice_scores2_1 = 0
            TP1 = 0
            TN1 = 0
            FP1 = 0
            FN1 = 0

            dice_scores_1c = 0
            dice_scores2_1c = 0
            TPc1 = 0
            TNc1 = 0
            FPc1 = 0
            FNc1 = 0        
            
            dice_scores_2 = 0
            dice_scores2_2 = 0
            TP2 = 0
            TN2 = 0
            FP2 = 0
            FN2 = 0
    
            # iterate over data
            dice_scores_g = 0
            TPg = 0
            TNg = 0
            FPg = 0
            FNg = 0
            
            dice_scores_c = 0
            TPc = 0
            TNc = 0
            FPc = 0
            FNc = 0 
    
            dice_scores_i = 0
            TPi = 0
            TNi = 0
            FPi = 0
            FNi = 0
            loop=tqdm(dataloader)
            for batch in loop:
                x, y11, y2 = batch
                x = x.to(device)
                y11 = y11.long().to(device)           
                y2 = y2.float().to(device)

                step += 1

                # forward pass
                if phase == 'train':
                    outputs11, outputs2 = model(x)
                    # calculate the loss 
                    loss1 = criterion1(outputs11, y11.squeeze(dim=1))

                    loss2 = criterion(outputs2.squeeze(dim=1), y2)
                    loss = loss1 + 0.7*loss2 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                
                else:
                    with torch.no_grad():
                        outputs11, outputs2 = model(x)
                        # calculate the loss 
                        loss1 = criterion1(outputs11, y11.squeeze(dim=1))
    
                running_loss += loss1
                
                ##########    
                preds1 = torch.argmax(outputs11, dim = 1)
                preds1= preds1.cpu().numpy().astype(int)
                yy1 = y11 
                yy1 = yy1.cpu().numpy().astype(int)
                num_correct1 += np.sum(preds1== yy1)            
                ### GGO
                TP1 += np.sum(( (preds1 == 1).astype(int) + (yy1 == 1).astype(int)) == 2)
                TN1 += np.sum(( (preds1 != 1).astype(int) + (yy1 != 1).astype(int)) == 2)
                FP1 += np.sum(( (preds1 == 1).astype(int) + (yy1 != 1).astype(int)) == 2)
                FN1 += np.sum(( (preds1 != 1).astype(int) + (yy1 == 1).astype(int)) == 2)
                num_pixels += preds1.size
    
                for idice in range(preds1.shape[0]):
                    predid1 = (preds1[idice] == 1).astype(int)
                    yyind1 = (yy1[idice] == 1).astype(int)
                    dice_scores_1 += (2 * (predid1 * yyind1).sum()) / (
                        (predid1  + yyind1).sum() + 1e-8
                    )  
                    
                     
                ### Consolidation    
                TPc1 += np.sum(( (preds1 == 2).astype(int) + (yy1 == 2).astype(int)) == 2)
                TNc1 += np.sum(( (preds1 != 2).astype(int) + (yy1 != 2).astype(int)) == 2)
                FPc1 += np.sum(( (preds1 == 2).astype(int) + (yy1 != 2).astype(int)) == 2)
                FNc1 += np.sum(( (preds1 != 2).astype(int) + (yy1 == 2).astype(int)) == 2)
     
                for idice in range(preds1.shape[0]): # Cons
                    predid1 = (preds1[idice] == 2).astype(int)
                    yyind1 = (yy1[idice] == 2).astype(int)
                    dice_scores_1c += (2 * (predid1 * yyind1).sum()) / (
                        (predid1  + yyind1).sum() + 1e-8
                    )
                #############
                loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                loop.set_postfix(loss=loss.item(), GGO_dice=dice_scores_1/preds1.shape[0],Cons_dice=dice_scores_1c/preds1.shape[0])
                del x; del y11; del y2
    
            epoch_loss = running_loss / len(dataloader.dataset)
            
            epoch_acc2_1 = (num_correct1/num_pixels)*100
            accbest = epoch_acc2_1
            epoch_dise_1 = dice_scores_1/len(dataloader.dataset)
            epoch_dise_1c = dice_scores_1c/len(dataloader.dataset)
    
            # GGO
            Spec1 = 1 - (FP1/(FP1+TN1))
            Sens1 = TP1/(TP1+FN1) # Recall
            Prec1 =  TP1/(TP1+FP1+ 1e-8)  
            F1score1 = TP1 / (TP1 + ((1/2)*(FP1+FN1))+ 1e-8)
            IoU1 = TP1 / (TP1+FP1+FN1)
    
            # Cons
            Spec1c = 1 - (FPc1/(FPc1+TNc1))
            Sens1c = TPc1/(TPc1+FNc1) # Recall
            Prec1c =  TPc1/(TPc1+FPc1+ 1e-8) 
            F1score1c = TPc1 / (TPc1 + ((1/2)*(FPc1+FNc1))+ 1e-8)
            IoU1c = TPc1 / (TPc1+FPc1+FNc1)
            print(f'Epoch [{epoch + 1}/{epochs}]:'," phase:",phase," epoch_loss:",round(epoch_loss.item(),3),"GOO_F1score:",round(F1score1,3),"GOO_IoU:",round(IoU1,3),"Cons_F1score:", round(F1score1c,3),"Cons_IoU:", round(IoU1c,3))
            ##########
          
            
            if phase == 'valid':
                if accbest > best_F1score:
                    best_F1score = accbest
                    torch.save(model.state_dict(), name_model_bestF1)
                    
            with open(training_tsx, "a") as f:
                print('Epoch {}/{}'.format(epoch, epochs - 1), file=f)
                print('-' * 10, file=f)                        
                print('{} Loss: {:.4f} Acc: {:.8f} Dise1: {:.8f} Dise2: {:.8f} IoU1: {:.8f} IoU2: {:.8f}'  \
                      ' F1: {:.8f} F12: {:.8f} Spec: {:.8f} Spec2: {:.8f} Sens: {:.8f} Sens2: {:.8f} Prec: {:.8f} Prec2: {:.8f}' \
                      .format(phase, epoch_loss, epoch_acc2_1, epoch_dise_1, epoch_dise_1c, \
                              IoU1,IoU1c, F1score1,F1score1c, Spec1,Spec1c, Sens1, Sens1c, Prec1, Prec1c), file=f)
                  
            
            train_loss.append(np.array(epoch_loss.detach().cpu())) if phase=='train' \
                else valid_loss.append(np.array(epoch_loss.detach().cpu()))
                
            train_acc.append(np.array(epoch_acc2_1)) if phase=='train' \
                else (valid_acc.append(np.array(epoch_acc2_1))) 
                      
            (train_dise1.append(np.array(epoch_dise_1)), train_dise2.append(np.array(epoch_dise_1c)))if phase=='train' \
                else (valid_dise1.append(np.array(epoch_dise_1)), \
                      valid_dise2.append(np.array(epoch_dise_1c)),
                      valid_Prec1.append(np.array(Prec1)),
                      valid_Prec2.append(np.array(Prec1c)),
                      valid_Spec1.append(np.array(Spec1)),
                      valid_Spec2.append(np.array(Spec1c)),
                      valid_Recall1.append(np.array(Sens1)),
                      valid_Recall2.append(np.array(Sens1c))
                      )
                    
                
            (train_IoU1.append(np.array(IoU1)), train_IoU2.append(np.array(IoU1c)))  if phase=='train' \
                else (valid_IoU1.append(np.array(IoU1)), \
                      valid_IoU2.append(np.array(IoU1c)))
                
            (train_F1score1.append(np.array(F1score1)), train_F1score2.append(np.array(F1score1c))) if phase=='train' \
                else (valid_F1score1.append((np.array(F1score1))), \
                      valid_F1score2.append((np.array(F1score1c))))
                
    torch.save(model.state_dict(), name_model_final)
    time_elapsed = time.time() - start
    with open(training_tsx, "a") as f:   
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), file=f)
                    
    ############################
    with open(training_tsx, "a") as f:
        
        ############################
        print('', file=f)
        print('Based on Accuracy', file=f)
        print('', file=f)
        print(np.max(valid_dise1), file=f)
        print(np.max(valid_dise2), file=f)
        print('Val', file=f)
        print('Best Val F1 score', file=f)
        print(np.max(valid_F1score1), file=f)
        print(np.max(valid_F1score2), file=f)
        print('Index of Best', file=f)
        print(name_model_bestF1, file=f)
        print(valid_acc.index(np.max(valid_acc)), file=f)

        
        
        print('GGO accBest', file=f)
        print("valid_acc:",valid_acc[valid_acc.index(np.max(valid_acc))], file=f)
        print("valid_F1score1:",valid_F1score1[valid_acc.index(np.max(valid_acc))], file=f)
        print("valid_dise1:",valid_dise1[valid_acc.index(np.max(valid_acc))], file=f)
        print("valid_IoU1:",valid_IoU1[valid_acc.index(np.max(valid_acc))], file=f)
        print("valid_Recall1:",valid_Recall1[valid_acc.index(np.max(valid_acc))], file=f)
        print("BestF1:",np.max(valid_F1score1),"epoch:",valid_F1score1.index(np.max(valid_F1score1)), file=f)
        print("BestRecall",np.max(valid_Recall1),"epoch:",valid_Recall1.index(np.max(valid_Recall1)), file=f)
        
        print('Consolidation accBest', file=f)
        print("valid_F1score2:",valid_F1score2[valid_acc.index(np.max(valid_acc))], file=f)
        print("valid_dise2:",valid_dise2[valid_acc.index(np.max(valid_acc))], file=f)
        print("valid_IoU2:",valid_IoU2[valid_acc.index(np.max(valid_acc))], file=f)
        print("valid_Recall2:",valid_Recall2[valid_acc.index(np.max(valid_acc))], file=f)
        print("BestF1:",np.max(valid_F1score2),"epoch:",valid_F1score2.index(np.max(valid_F1score2)), file=f)
        print("BestRecall",np.max(valid_Recall2),"epoch:",valid_Recall2.index(np.max(valid_Recall2)), file=f)
        
        print('-' * 10, file=f)
        print('val Results', file=f)
        print('val_acc', valid_acc, file=f)
        print('val_F1score1', valid_F1score1, file=f)
        print('val_dise1',valid_dise1, file=f)
        print('val_IoU1',valid_IoU1, file=f)
        print('val_F1score2', valid_F1score2, file=f)
        print('val_dise2',valid_dise2, file=f)
        print('val_IoU2',valid_IoU2, file=f) 
        print('-' * 10, file=f)        

        print('-' * 10, file=f)
        print('train Results', file=f)
        print('train_acc', train_acc, file=f)
        print('train_F1score1', train_F1score1, file=f)
        print('train_dise1',train_dise1, file=f)
        print('train_IoU1',train_IoU1, file=f)
        print('train_F1score2', train_F1score2, file=f)
        print('train_dise2',train_dise2, file=f)
        print('train_IoU2',train_IoU2, file=f)        



        acc_mean.append(np.max(valid_acc))
        F1_mean1.append(valid_F1score1[valid_F1score2.index(np.max(valid_F1score2))])
        dise_mean1.append(valid_dise1[valid_F1score2.index(np.max(valid_F1score2))])
        IoU_mean1.append(valid_IoU1[valid_F1score2.index(np.max(valid_F1score2))])
        Recall_mean1.append(valid_Recall1[valid_F1score2.index(np.max(valid_F1score2))])
        Prec_mean1.append(valid_Prec1[valid_F1score2.index(np.max(valid_F1score2))])
        Spec_mean1.append(valid_Spec1[valid_F1score2.index(np.max(valid_F1score2))])

        F1_mean2.append(valid_F1score2[valid_F1score2.index(np.max(valid_F1score2))])
        dise_mean2.append( valid_dise2[valid_F1score2.index(np.max(valid_F1score2))])
        IoU_mean2.append(valid_IoU2[valid_F1score2.index(np.max(valid_F1score2))])
        Recall_mean2.append(valid_Recall2[valid_F1score2.index(np.max(valid_F1score2))])
        Prec_mean2.append(valid_Prec2[valid_F1score2.index(np.max(valid_F1score2))])
        Spec_mean2.append(valid_Spec2[valid_F1score2.index(np.max(valid_F1score2))])
                
    f.close()
    

stdacc=np.std(acc_mean[:5])
std1 = np.std(F1_mean1[:5])
std2 = np.std(dise_mean1[:5])
std3 = np.std(IoU_mean1[:5])
std4=np.std(Recall_mean1[:5])
std5=np.std(Prec_mean1[:5])
std6=np.std(Spec_mean1[:5])

std12 = np.std(F1_mean2[:5])
std22 = np.std(dise_mean2[:5])
std32 = np.std(IoU_mean2[:5])
std42=np.std(Recall_mean2[:5])
std52=np.std(Prec_mean2[:5])
std62=np.std(Spec_mean2[:5])

training_tsx = model_spR + '/' + 'mean' + '.txt'
acc_mean.append(np.mean(acc_mean[:5]))
F1_mean1.append(np.mean(F1_mean1[:5]))
dise_mean1.append(np.mean(dise_mean1[:5]))
IoU_mean1.append(np.mean(IoU_mean1[:5]))
Recall_mean1.append(np.mean(Recall_mean1[:5]))
Spec_mean1.append(np.mean(Spec_mean1[:5]))
Prec_mean1.append(np.mean(Prec_mean1[:5]))

F1_mean2.append(np.mean(F1_mean2[:5]))
dise_mean2.append(np.mean(dise_mean2[:5]))
IoU_mean2.append(np.mean(IoU_mean2[:5]))
Recall_mean2.append(np.mean(Recall_mean2[:5]))
Spec_mean2.append(np.mean(Spec_mean2[:5]))
Prec_mean2.append(np.mean(Prec_mean2[:5]))

acc_mean.append(stdacc)
F1_mean1.append(std1)
dise_mean1.append(std2)
IoU_mean1.append(std3)
Recall_mean1.append(std4)
Prec_mean1.append(std5)
Spec_mean1.append(std6)

F1_mean2.append(std12)
dise_mean2.append(std22)
IoU_mean2.append(std32)
Recall_mean2.append(std42)
Prec_mean2.append(std52)
Spec_mean2.append(std62)


with open(training_tsx, "a") as f:
    print("acc",acc_mean,file=f)
    print('GGO', file=f)
    print('F1_mean', F1_mean1, file=f)
    print('dise_mean', dise_mean1, file=f)
    print('IoU_mean', IoU_mean1, file=f)
    print("Recall",Recall_mean1,file=f)
    print("Spec",Spec_mean1,file=f)
    print("Precc", Prec_mean1, file=f)
    
    print('Cons', file=f)
    print('F1_mean', F1_mean2, file=f)
    print('dise_mean', dise_mean2, file=f)
    print('IoU_mean', IoU_mean2, file=f)
    print("Recall",Recall_mean2,file=f)
    print("Spec",Spec_mean2,file=f)
    print("Precc", Prec_mean2, file=f)
f.close()
