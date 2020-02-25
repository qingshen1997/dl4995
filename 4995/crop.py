from torchvision import models
from torch import nn
from torch import optim
import os
import numpy as np
import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
from skimage import util
import random

def gasuss_noise(image):
    image = np.array(image)
    case = random.randint(1, 3)
    if case == 1:
        image = util.random_noise(image, mode='gaussian')
    elif case == 2:
        image = util.random_noise(image, mode='s&p')
    image = np.float32(image)
    return image


transform = transforms.Compose(
    [transforms.Resize((1024, 1024)),transforms.RandomCrop(1024, padding=4),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomRotation(30, resample=False, expand=False, center=None),
    torchvision.transforms.Lambda(lambda img: gasuss_noise(img)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


transform_test = transforms.Compose(
    [transforms.Resize((1024, 1024)),transforms.CenterCrop(1024),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.ImageFolder(root="train/"
                                         ,transform=transform)




testset = torchvision.datasets.ImageFolder(root="test/", transform=transform_test)
names = []
for i in range(len(testset.imgs)):
    names.append(testset.imgs[i][0].split('/')[-1])
    names[i] = names[i].replace(".jpg", "")
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                        shuffle=False, num_workers=2)


resnet_model = models.resnet50(pretrained=True)
class_num = 3
channel_in =resnet_model.fc.in_features
resnet_model.fc = nn.Linear(channel_in,class_num)
for para in list(resnet_model.parameters())[:-2]:
    para.requires_grad = False



net = resnet_model
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

for epoch in range(50):
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
        if i % 2 == 1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2))
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
i = 0
with torch.no_grad():
    for data in testloader:
        images,_ = data
        if torch.cuda.is_available():
            images = images.to(device)
        outputs = net(images)
        outputs = F.softmax(outputs)

        outputs = outputs.numpy().tolist()[0]
        for j in range(len(outputs)):
            outputs[j] = str(outputs[j])
        outputs[0], outputs[2] = outputs[2], outputs[0]

        new_row = [names[i]] + outputs
        preds.append(new_row)
        i += 1
test_res=pd.DataFrame(columns=['ID','leaf_rust','stem_rust','healthy_wheat'],data=preds)
test_res.to_csv('submissionfile.csv', index=0)
