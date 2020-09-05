import numpy as np
import pandas as pd
import os
from PIL import Image
import seaborn as sns
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
import adabound

from sklearn import metrics

expt = 3

def metric(ori, pred):
    lis_ori = []
    lis_pred = []
    for i in ori:
        lis_ori.append(i.item())
    for i in pred:
        lis_pred.append(i.item())
    fpr, tpr, thresholds = metrics.roc_curve(lis_ori, lis_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print(fpr, tpr, thresholds, roc_auc)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#     plt.title('ROC for digit=%d class' % 2)
    plt.legend(loc="lower right")
    plt.savefig("ROC"+str(expt)+".png")

    #plt.show()



#######DATA Transforms
### Normalize the data
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

data_dir = 'val/'
new_data_dir = 'synthetic_data/'
train_data = datasets.ImageFolder(new_data_dir,
                                    transform=data_transforms['train'])
val_data = datasets.ImageFolder(data_dir,
                                    transform=data_transforms['val'])

train_loader=DataLoader(dataset=train_data,batch_size=64,num_workers=1,shuffle=True)
val_loader=DataLoader(dataset=val_data,batch_size=64,num_workers=1,shuffle=True)

dataloaders={
    'train':train_loader,
    'val':val_loader
}

dataset_sizes={
    'train':len(train_data),
    'val':len(val_data)
}

## 100s of tirads (same architecture)
## videos -- input (nodules present or not.)
import numpy as np

train_ac = []
train_pred = []
val_ac = []
val_pred = []

train_prob = []
val_prob = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
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
                model.eval()  # Set model to evaluate mode

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

                    # if (phase == 'train'):
                    #     train_pred.extend(preds)
                    #     train_ac.extend(labels.view_as(preds))
                    #     train_prob.extend(np.exp(outputs.detach().numpy()[:, 1]))

                    if (phase == 'val'):
                        val_pred.extend(preds)
                        val_ac.extend(labels.view_as(preds))
                        val_prob.extend(np.exp(outputs.detach().numpy()[:, 1]))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                #                 print(loss)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': epoch_acc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), }


            if (phase == 'val'):
                metric(val_ac, val_prob)
                print(metrics.classification_report(val_ac, val_pred))


            # Save best model
            if (epoch_acc > old_acc):
                print(old_acc, epoch_acc)
                old_acc = epoch_acc
                file = "model_synthetic"+str(expt)+".pth"
                torch.save(checkpoint, file)

                #if (phase == 'val'):
                    #metric(val_ac, val_prob)
                    #print(metrics.classification_report(val_ac, val_pred))

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
model=models.resnet18(pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
for param in model.parameters():
    param.requires_grad = False


#RESNET
model.fc = nn.Sequential(
               nn.Linear(512, 2))#,
               #nn.ReLU(inplace=True),
               #nn.Linear(128, 2)).to(device)

print(model)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer = optim.Adam(model.fc.parameters(), lr=0.005)
#optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)
optimizer = adabound.AdaBound(model.parameters(), lr=0.005, final_lr=0.009)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=30)
