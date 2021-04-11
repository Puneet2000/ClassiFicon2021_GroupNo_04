import numpy as np
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torchvision.utils as vutils
from scipy.stats import truncnorm
import json 
import glob
import os
import random
from PIL import Image
import numpy as np
import torchvision.models as models
import torchvision.transforms.functional as Fn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import pandas as pd
torch.backends.cudnn.benchmark=True
import torchvision
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(0)
# Arguments
parser = argparse.ArgumentParser(
    description='Train ICP.'
)
parser.add_argument('--exclude', type=str, default='negative smile',help='Path to cofig file.')
parser.add_argument('--epochs', type=int, default=100, help='Path to cofig file.')
parser.add_argument('--batch_size', type=int, default=256, help='Path to cofig file.')

args = parser.parse_args()


# print(images.shape)

class TripletSmileDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=True):
        super(TripletSmileDataset, self).__init__()
        if train:
            self.df = pd.read_csv('./train.csv', header=None)
        else:
            self.df = pd.read_csv('./test.csv', header=None)

        self.df_pos = self.df[self.df[1]=='positive smile']
        self.df_no = self.df[self.df[1]=='NOT smile']
        self.df_neg = self.df[self.df[1]=='negative smile']


        self.transform = transform

    def __len__(self):
        return max(max(len(self.df_pos), len(self.df_no)), len(self.df_neg))
    def __getitem__(self, idx):

        idx_pos = idx%len(self.df_pos)
        idx_no = idx%len(self.df_no)
        idx_neg = idx%len(self.df_neg)

        row_pos = self.df_pos.iloc[idx_pos]
        image_pos = self.transform(Image.open(os.path.join('./happy_images', row_pos[0] + '.jpg')).convert('RGB'))

        row_no = self.df_no.iloc[idx_no]
        image_no = self.transform(Image.open(os.path.join('./happy_images', row_no[0] + '.jpg')).convert('RGB'))

        row_neg = self.df_neg.iloc[idx_neg]
        image_neg = self.transform(Image.open(os.path.join('./happy_images', row_neg[0] + '.jpg')).convert('RGB'))

        return image_pos, image_no, image_neg, 2, 1, 0

class SmileDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=True):
        super(SmileDataset, self).__init__()
        if train:
            self.df = pd.read_csv('./train.csv', header=None)
        else:
            self.df = pd.read_csv('./test.csv', header=None)

        self.labels = [ 'negative smile', 'NOT smile', 'positive smile']

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row[0]
        label = self.labels.index(row[1])
        image_path = os.path.join('./happy_images', image_id + '.jpg')
        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        return image, label

def val(loader, model):
    eps = 0.1
    model.eval()
    correct, total = 0, 0
    losses = []
    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            x,y = x.cuda(), y.cuda()
            x[:,:,:150,:] = -1.
            score = model(x)
            pred = torch.argmax(score, 1)
            loss = F.cross_entropy(score, y)
            losses.append(loss.item())

            correct += (pred==y).sum().item()
            total += pred.size(0)

    return (100.*correct/total), np.mean(losses)

transform = transforms.Compose([transforms.Resize((224,224)),
								transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

trainset = SmileDataset(transform=transform, train=True)
triplet_trainset = TripletSmileDataset(transform=transform, train=True)
triplet_train_loader = torch.utils.data.DataLoader(
        triplet_trainset,
        batch_size=64,
        shuffle=True,
        num_workers=2)
train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)

testset = SmileDataset(transform=transform, train=False)
test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)

  
resnet50 = models.resnet18(pretrained=True)  
num_ftrs = resnet50.fc.in_features
# resnet50.fc = nn.Linear(num_ftrs, 14)

# resnet50.load_state_dict(torch.load('./celebA_resnet50.pth')['model'])
resnet50.fc = nn.Linear(num_ftrs, 3)
for name, param in resnet50.named_parameters():
    if 'fc' in name or 'bn' in name:
        # print('Enabling ', name)
        param.requires_grad = True
    else:
        param.requires_grad = False

resnet50 = nn.DataParallel(resnet50)
resnet50 = resnet50.cuda()
# resnet50.load_state_dict(torch.load('./?model.pth'))
resnet50.eval()

optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.001)
scores = []
best_val = 0.
d = {'train_acc':[], 'train_loss':[], 'test_acc':[], 'test_loss':[]}
for epoch in range(args.epochs):
    losses = []
    correct, total = 0, 0
    correct_, total = 0, 0
    resnet50.train()
    for i, (x_pos, x_no, x_neg, y_pos, y_no, y_neg) in enumerate(triplet_train_loader):
        # print(i)
        x = torch.cat([x_pos, x_no, x_neg],0)
        y = torch.cat([y_pos, y_no, y_neg],0)
        x,y = x.cuda(), y.cuda()
        x[:,:,:150,:] = -1.
        torchvision.utils.save_image(x[:10].cpu(), './x.png', normalize=True)

        idx= torch.randperm(x.size(0))
        lam = np.random.beta(1,1)
        x_m = lam*x + (1-lam)*x[idx]
        score = resnet50(x_m)
        loss = lam*F.cross_entropy(score, y) + (1-lam)*F.cross_entropy(score, y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i%20==0:
        #     print(i, loss.item())

    train_acc, train_loss = val(train_loader, resnet50)
    test_acc, test_loss = val(test_loader, resnet50)
    d['train_acc'].append(train_acc)
    d['train_loss'].append(train_loss)
    d['test_acc'].append(test_acc)
    d['test_loss'].append(test_loss)
    if test_acc> best_val:
        best_val =  test_acc
        torch.save(resnet50.state_dict(), './model.pth')

    print('Epoch {}, Train Acc {:.3f}, Train Loss {:.3f}, Test Acc {:.3f}, Test Loss {:.3f}'.format(epoch, train_acc, train_loss, test_acc, test_loss))

np.savez('./dictionary.npz', d)