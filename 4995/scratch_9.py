import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
# Import dataset related API
import torchvision
import torchvision.transforms as transforms

# Import common neural network API in pytorch
import torch.nn as nn
import torch.nn.functional as F

# Import optimizer related API
import torch.optim as optim

# Fix the random seed of pytorch related function
torch.manual_seed(0)

# Fix the random seed of numpy related function
np.random.seed(0)
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
# Check device, using gpu 0 if gpu exist else using cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset

transform = transforms.Compose(
    [transforms.Resize((512, 512)),transforms.RandomCrop(512),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose(
    [transforms.Resize((512, 512)),transforms.CenterCrop(512),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.ImageFolder(root="train/"
                                         ,transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)





testset = torchvision.datasets.ImageFolder(root="test/", transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                        shuffle=False, num_workers=2)
names = []
for i in range(len(testset.imgs)):
    names.append(testset.imgs[i][0].split('/')[-1])
    names[i] = names[i].replace(".jpg", "")
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 12, 5)
        self.conv3 = nn.Conv2d(12, 16, 5)
        self.conv4 = nn.Conv2d(16, 24, 5)
        self.fc1 = nn.Linear(24 * 28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 24*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.008, momentum=0.9)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def saver(model, optimizer, ep, \
          lr_scheduler=None, save_path='./checkpoint.pt'):

    checkpoint = {'model': model.state_dict(),\
                  'optimizer': optimizer.state_dict(),\
                  'start_ep': ep + 1}
    # Other stuff you want to do...
    if lr_scheduler is not None:
        checkpoint = {'model': model.state_dict(),\
                    'optimizer': optimizer.state_dict(),\
                    'lr_scheduler': lr_scheduler.state_dict(),\
                    'start_ep': ep + 1}

    torch.save(checkpoint, save_path)

def loader(model, optimizer, \
           lr_scheduler=None, load_path="./checkpoint.pt"):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_ep = checkpoint['start_ep']
    # Other stuff you want to do..
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        return model, optimizer, lr_scheduler, start_ep
    else:
        return model, optimizer, start_ep

for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        if torch.cuda.is_available():
            inputs = inputs.to(device)
        # zero the parameter gradients: Clean the gradient caclulated in the previous iteration
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Calculate gradient of matrix with requires_grad = True
        loss.backward()

        # Apply the gradient calculate from last step to the matrix
        optimizer.step()
        # Add 1 more iteration count to learning rate scheduler
        #lr_scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i % 16 == 15:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 16))
            running_loss = 0.0

print('Finished Training')
saver(model=net, optimizer=optimizer, lr_scheduler=lr_scheduler, ep=2)

correct = 0
total = 0
# Switch some layers (e.g., batch norm, dropout) to evaluation mode
net.eval()
# Turn off the autograd to save memory usage and speed up
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the train images: %f %%' % (
    100 * float(correct) /float(total)))



##test
preds = []
test_count = 610

with torch.no_grad():
    for data in testloader:
        images,_ = data
        if torch.cuda.is_available():
            images = images.to(device)
        outputs = net(images)
        outputs = F.softmax(outputs)
        outputs = outputs.numpy().tolist()
        preds.append(outputs[0])
    print(preds)
