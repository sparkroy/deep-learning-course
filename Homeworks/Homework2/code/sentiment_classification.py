import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import math
from collections import OrderedDict


class text_classification(nn.Module):
    # You will implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # Since the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3, 
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self):
        self.conv = nn.Sequential(
            nn.
        )
        self.fc = nn.Sequential(
            # TODO: fully-connected layer (64->64)
            # TODO: fully-connected layer (64->10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            
            # print statistics
            # running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    embedding = False
    trainset = extract_tokens('data/train.txt', embedding)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True)

    testset = extract_tokens('data/test.txt', embedding)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)
    
    devset = extract_tokens('data/dev.txt', embedding)
    valloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)
    
    net = VGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)
    

if __name__== "__main__":
    main()
   
