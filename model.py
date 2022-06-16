import random
import pandas as pd
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import os
from torchmetrics import ConfusionMatrix
import argparse

class DataPro:
    transforms_train = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    train_path = os.path.join(os.getcwd(), "training")
    test_path = os.path.join(os.getcwd(), "test")
    image_data_train = ImageFolder(train_path, transform=transforms_train)
    image_data_test = ImageFolder(test_path, transform=transforms_train)

    random.shuffle(image_data_train.samples)
    random.shuffle(image_data_test.samples)

    classes_idx = image_data_train.class_to_idx
    classes = len(image_data_train.classes)
    len_train_data = len(image_data_train)
    len_test_data = len(image_data_test)


def get_labels():
    train_labels = []
    test_labels = []
    for i in DataPro.image_data_train.imgs:
        train_labels.append(i[1])

    for j in DataPro.image_data_test.imgs:
        test_labels.append(j[1])
    return train_labels, test_labels


labels_train, labels_test = get_labels()
train_loader = DataLoader(dataset=DataPro.image_data_train, batch_size=100)
test_loader = DataLoader(dataset=DataPro.image_data_test, batch_size=100)




class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Visio_block(nn.Module):
    def __init__(self,in_channels, exp_1x1, ipa_3x3, out_3x3):
        super(Visio_block, self).__init__()
        self.edge1 = nn.Sequential(conv_block(in_channels, exp_1x1, kernel_size=(1, 1)), nn.MaxPool2d(2))
        self.edge2 = nn.Sequential(conv_block(exp_1x1, ipa_3x3, kernel_size=(1, 1)), nn.MaxPool2d(2))
        self.edge3 = nn.Sequential(conv_block(ipa_3x3, out_3x3, kernel_size=(1, 1)), nn.AvgPool2d(5))
        self.edge4 = nn.Sequential(Flatten(), nn.Linear(5*5*out_3x3, 2))

    def forward(self, x):
        x = self.edge1(x)
        x = self.edge2(x)
        x = self.edge3(x)
        x = self.edge4(x)
        return x


model = Visio_block(3, 64, 64, 128)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


def train(epochs):
    model.train()
    losses = []
    for epoch in range(1, epochs + 1):
        print("epoch #", epoch)
        current_loss = 0.0
        for feature, label in train_loader:
            x = Variable(feature, requires_grad=False).float()
            y = Variable(label, requires_grad=False).long()
            optimizer.zero_grad()
            y_pred = model(x)
            correct = y_pred.max(1)[1].eq(y).sum()
            print("number of correct items classified: ", correct.item(),"/", 100)
            loss = criterion(y_pred, y)
            print("loss: ", loss.item())
            current_loss += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(current_loss)
    return losses


def test():
    model.eval()
    with torch.no_grad():
        for feature, label in test_loader:
            pred = model(feature)
            print("acc: ", accuracy_score(labels_test, pred.max(1)[1].data.numpy()) * 100)
            loss = criterion(pred, label)
            print("loss: ", loss.item())

def get_confusion_matrix():
    model.eval()
    with torch.no_grad():
        for feature, label in test_loader:
            prediction = model(feature)
            conf_mat = ConfusionMatrix(num_classes=2)
            print("Confision Matrix", conf_mat(prediction, label))
            print("TN=", conf_mat(prediction, label)[0][0])
            print("FN=", conf_mat(prediction, label)[1][1])
            print("TP=", conf_mat(prediction, label)[0][1])
            print("FP=", conf_mat(prediction, label)[1][0])


def active_tenserboard():
    writer = SummaryWriter()
    writer.flush()


train(3)
test()
get_confusion_matrix()
