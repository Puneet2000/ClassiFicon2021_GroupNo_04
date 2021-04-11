import os
import pdb

from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms.transforms import ColorJitter, RandomHorizontalFlip
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import dataloader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-lr', type=float, default=1e-2, help='leanring rate')
    parser.add_argument('-n', '--num_epochs', type=int, default=60, help='num epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-l2', type=float, default=1e-4, help='l2 decay')
    parser.add_argument('-m', '--milestones', type=int, default=[20, 40], nargs='+', help='milestones for lr-decay')
    parser.add_argument('--sampler', action='store_true', help='use sampler')
    parser.add_argument('-gamma', type=float, default=0.1, help='lr decay gamma')
    parser.add_argument('-cuda', type=int, default=0, help='gpu index')

    return parser.parse_args()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detech_smile(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return image
    elif len(faces) == 1:
        x, y, w, h = faces[0]
        smiles = smile_cascade.detectMultiScale(gray[y:y+h, x+x+w], 1.8, 20)
        if len(smiles) == 0:
            return image
        elif len(smiles) == 1:
            sx, sy, sw, sh = smiles[0]
            image = image.crop((sx, sy, sx+sw, sy+sh))
            return image
        else:
            image = image.crop((x, y, x+w, y+h))
            return image
    else:
        return image


class SmileDataset(data.Dataset):
    def __init__(self, df, label_map, transform) -> None:
        super(SmileDataset, self).__init__()
        self.df = df
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row[0]
        label = self.label_map[row[1]]
        image_path = osp.join('./happy_images', image_id + '.jpg')
        image = Image.open(image_path)
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = self.transform(image)

        return image, label

class NormalBlock(nn.Module):
    def __init__(self, ch) -> None:
        super(NormalBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(ch),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.net(x)
        return out + x

class ReductionBlock(nn.Module):
    def __init__(self, inch, outch):
        super(ReductionBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outch),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        out = self.net(x)
        return self.pool(out)

class SmileNet(nn.Module):

    def __init__(self, num_classes):
        super(SmileNet, self).__init__()

        self.width = 8
        # add dropout
        def reduce_block(inch, outch):
            conv = nn.Sequential(
                nn.Conv2d(inch, outch, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(outch),
                nn.ReLU(),
                nn.MaxPool2d((2, 2))
            )
            return conv

        def normal_block(inch, outch):
            conv = nn.Sequential(
                nn.Conv2d(inch, outch, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(outch),
                nn.ReLU(),
            )

            return conv

        self.conv = nn.Sequential(
            reduce_block(3, 2 * self.width), 
            # normal_block(2 * self.width, 2 * self.width),
            reduce_block(2 * self.width, self.width), 
            normal_block(self.width, self.width),
            reduce_block(self.width, self.width), 
            normal_block(self.width, self.width),
            reduce_block(self.width, self.width)
        )
        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, self.width, kernel_size=3, padding=1, stride=1),
        #     NormalBlock(self.width),
        #     ReductionBlock(self.width, 2 * self.width),
        #     NormalBlock(2 * self.width),
        #     ReductionBlock(2 * self.width, 4 * self.width),
        #     NormalBlock(4 * self.width),
        #     ReductionBlock(4 * self.width, 2 * self.width),
        #     NormalBlock(2 * self.width),
        #     ReductionBlock(2 * self.width, self.width)
        # )
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * self.width, 100),
            nn.Linear(100, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):

        out = self.conv(x)
        out = out.view(-1, 8 * 8 * self.width)
        out = self.fc(out)

        return out

def train(model, dataloader, criterion, optimizer, device):
    model.train()

    # epoch_iters = len(dataloader)
    for i, (image, truth) in enumerate(dataloader):
        image, truth = image.to(device), truth.to(device)

        optimizer.zero_grad()

        with torch.enable_grad():
            output = model(image)
            loss = criterion(output, truth)
            loss.backward()
            optimizer.step()
            # scheduler.step(epoch - 1 + i / epoch_iters)
            # debug utils
            # _, preds = torch.max(output, 1)
            # corrects = torch.mean((preds == truth.data).float())
            # print(corrects.item(), loss.item())


def test(model, dataloader, criterion, device):
    model.eval()
    running_corrects = 0
    running_loss = 0
    for image, truth in dataloader:
        image, truth = image.to(device), truth.to(device)

        with torch.no_grad():
            output = model(image)
            loss = criterion(output, truth)
            _, pred = torch.max(output, 1)
            running_corrects += torch.sum(pred == truth.data)
            running_loss += loss.item() * image.size(0)

    dataset_size = len(dataloader.dataset)

    return running_loss / dataset_size, running_corrects.double() / dataset_size

def remove_not_smile(df):

    return df[df[1] != 'negative smile']

def print_baseline(df):

    df = df[1]
    counts = df.value_counts()
    total = sum(counts)
    for classname, classcount in counts.iteritems():
        print(classname, 'class:', classcount / total)

def count_classes(df):
    return len(pd.unique(df[1]))

def get_class_weight_sampler(df):

    labels_unique, counts = np.unique(df[1], return_counts=True)
    class_weights = {l: sum(counts) / c for (l, c) in zip(labels_unique, counts)}
    example_weights = [class_weights[e] for e in df[1]]
    sampler = WeightedRandomSampler(example_weights, len(train_df))
    return sampler

def fine_tune(model):
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, 'classifier'):
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3),
        )
    elif hasattr(model, 'fc'):
        model.fc.requires_grad = True

if __name__ == '__main__':

    args = parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    train_csv_path = 'train.csv'
    test_csv_path = 'test.csv'
    assert osp.exists(train_csv_path)
    assert osp.exists(test_csv_path)

    label_map = {
        'positive smile': 0,
        'NOT smile': 1,
        'negative smile': 2
    }

    # add all possible transforms
    train_transform = transforms.Compose([
        # detech_smile,
        transforms.Resize((256, 256)),
        transforms.RandomRotation(30),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_df = pd.read_csv(train_csv_path, header=None)
    test_df = pd.read_csv(test_csv_path, header=None)

    # train_df = remove_not_smile(train_df)
    # test_df = remove_not_smile(test_df)
    num_classes = count_classes(train_df)
    print('train baseline')
    print_baseline(train_df)

    print('train_df:', train_df.shape)
    print('test_df:', test_df.shape)


    train_dataset = SmileDataset(train_df, label_map, train_transform)
    test_dataset = SmileDataset(test_df, label_map, test_transform)

    sampler = None
    if args.sampler:
        sampler = get_class_weight_sampler(train_df)

    num_workers = 1
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, sampler=sampler)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device(args.cuda)
    criterion = nn.CrossEntropyLoss()
    # model = SmileNet(num_classes).to(device)
    model = models.vgg11_bn(pretrained=True)
    fine_tune(model)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, T_mult=2)

    for epoch in range(1, args.num_epochs + 1):
        print('epoch', epoch)
        train(model, train_dataloader, criterion, optimizer, device)
        scheduler.step()
        # eval
        # train
        loss, acc = test(model, train_dataloader, criterion, device)
        print('train acc: {:.3f}'.format(acc))
        # test
        loss, acc = test(model, test_dataloader, criterion, device)
        print('test acc: {:.3f}'.format(acc))