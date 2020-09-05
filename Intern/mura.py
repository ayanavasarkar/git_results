import numpy as np
import pandas as pd
import os
from PIL import Image
import seaborn as sns

train_df = pd.read_csv('train+labels.csv')
val_df = pd.read_csv('val+labels.csv')


import torch
import torch.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Normalize,CenterCrop,Resize,Compose
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = 'MURA-v1.1/data/'
train_data = datasets.ImageFolder(data_dir + 'train',  
                                    transform=data_transforms['train'])                                       
val_data = datasets.ImageFolder(data_dir + 'val', 
                                    transform=data_transforms['val'])



train_loader=DataLoader(dataset=train_data,batch_size=32,num_workers=1,shuffle=True)
val_loader=DataLoader(dataset=val_data,batch_size=32,num_workers=1,shuffle=True)

dataloaders={
    'train':train_loader,
    'val':val_loader
}

dataset_sizes={
    'train':len(train_data),
    'val':len(val_data)
}

train_acc = []
val_acc = []
old_acc = 0.0


import numpy as np
train_ac = []
train_pred = []
val_ac = []
val_pred = []

train_prob = []
val_prob = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    old_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print(phase, epoch)
                model.train()  # Set model to training mode
            else:
                print(phase, epoch)
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

		    #if(phase=='train'):
                        #train_pred.extend(preds)
                        #train_ac.extend(labels.view_as(preds))
                        #train_prob.extend(np.exp(outputs.detach().numpy()[:, 1]))
                        
                    #elif(phase=='val'):
                        #val_pred.extend(preds)
                        #val_ac.extend(labels.view_as(preds))
                        #val_prob.extend(np.exp(outputs.detach().numpy()[:, 1]))


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                #print(loss)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_acc.append(epoch_acc)
            else:
                val_acc.append(epoch_acc)


            checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': epoch_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),}

            #Save best model
            if(epoch_acc>old_acc):
                print("Model Saving")
                print(old_acc, epoch_acc)
                old_acc = epoch_acc
                torch.save(checkpoint, 'test2.pth')
		

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
print(model)


## RESNET
model.fc = nn.Sequential(
               nn.Linear(2048, 1024),
               nn.ReLU(inplace=True),
	       nn.Linear(1024, 128),
	       nn.ReLU(inplace =True),
               nn.Linear(128, 2)).to(device)


#model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.fc.parameters(), lr=0.05)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=10)




print(train_acc, val_acc)

df_train = pd.DataFrame(train_acc)
#df_train.to_csv('train_acc_resnet50_gradient.csv', index=False)

df_val = pd.DataFrame(val_acc)
#df_val.to_csv('val_acc_resnet50_gradient.csv', index=False)
