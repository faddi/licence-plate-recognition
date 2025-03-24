import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from glob import glob
import math
import random


def getRandomSampleImageAndLabel(path):
    image_path_list = glob(path + 'images/*.png')
    image_path_list1 = image_path_list.sort()
    label_path_list = glob(path + 'labels/*.txt')
    label_path_list1 = label_path_list.sort()
    lenth_image_list = len(image_path_list)
    rand_index = random.randint(0,lenth_image_list-1)
    print(image_path_list[rand_index])
    print(label_path_list[rand_index])

    image = Image.open(image_path_list[rand_index])
    label = np.loadtxt(label_path_list[rand_index])
    return image, label

def getBoundingBoxCoords(label, size):
    middleX, middleY, width, height = scalingOfLabel(label, size)
    halfWidth = width / 2
    halfHeight = height / 2
    topLeftX = int(middleX - halfWidth)
    topLeftY = int(middleY - halfHeight)
    bottomRightX = int(middleX + halfWidth)
    bottomRightY = int(middleY + halfHeight)
    box = [topLeftX, topLeftY, bottomRightX, bottomRightY]
    return box

def scalingOfLabel(label, size):
    x,y = size
    boundingBoxMiddleInX = label[0] * x
    boundingBoxMiddleInY = label[1] * y
    width = label[2] * x
    height = label[3] * y
    return boundingBoxMiddleInX, boundingBoxMiddleInY, width, height


data = np.loadtxt("Data/Train Data.csv",
                 delimiter=",", dtype=np.float64, skiprows=1)

data_dir_sample = 'Data/test/images/Cars2.png'

test = Image.open(data_dir_sample)
sample_img = plt.imread(data_dir_sample)

data_dir_train = 'Data/train/'
data_dir_test = 'Data/test/'

im, label = getRandomSampleImageAndLabel(data_dir_train)


boundingBoxCoords = getBoundingBoxCoords(label[1:], im.size)
#draw = ImageDraw.Draw(im)
#draw.rectangle(boundingBoxCoords)



class LicencePlateData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.xData = glob(data_dir + 'images/*.png')
        self.yData = glob(data_dir + 'labels/*.txt')
        self.xData.sort()
        self.yData.sort()
        self.transform = transform
        self.n_samples = len(self.xData)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        feature_dir = self.xData[index]
        label_dir = self.yData[index]
        print(feature_dir)
        print(label_dir)
        feature = torch.from_numpy(plt.imread(feature_dir)[:,:,:3])
        
        x_scaler = 128 / feature.shape[0]
        y_scaler = 128 / feature.shape[1]
        #print(feature.shape)
        feature = feature.permute(2,0,1)
        feature = self.transform(feature)

        label = torch.from_numpy(np.loadtxt(label_dir)[1:])

        
        return feature.float(), label.float()
    
    def getPath(self, index):
        return self.xData[index], self.yData[index]


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # input: 3 x 128 x 128
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 32 x 128 x 128
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 64 x 128 x 128
            nn.MaxPool2d(2, 2),
            # output: 64 x 64 x 64
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output 128 x 64 x 64
            nn.MaxPool2d(2, 2),
            # output 128 x 32 x 32
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 256 x 32 x 32
            nn.MaxPool2d(2, 2),
            # output: 256 x 16 x 16


            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 512 x 16 x 16
            nn.MaxPool2d(2, 2),
            # output: 512 x 8 x 8
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 1024 x 8 x 8
            nn.MaxPool2d(2,2),
            # output: 1024 x 4 x 4

            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)
    

training_data = LicencePlateData(data_dir_train, transform=transforms.Resize((128,128)))

test_data = LicencePlateData(data_dir_test, transform=transforms.Resize((128,128)))


feature, label = training_data[66]
num_epochs = 5
batch_size = 10
learning_rate = 0.001
nr_of_samples = training_data.n_samples
nr_of_iterations = math.ceil(batch_size/nr_of_samples)
nr_of_outputs = label.shape[0]

training_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)



model = NeuralNet()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

n_total_steps = len(training_loader)
print(n_total_steps)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(training_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item()}')

# test
with torch.no_grad():
    test_loss= 0
    for test_images, test_labels in test_loader:
        test_outputs = model(test_images)
        test_loss += criterion(test_outputs, test_labels)
        rand_test_img_sample = test_images[0]
        rand_test_img_label = test_labels[0]
        rand_test_output = test_outputs[0]

        

    bb_coords_label = getBoundingBoxCoords(rand_test_img_label, (128, 128))
    bb_coords_output = getBoundingBoxCoords(rand_test_output, (128, 128))
    bb_coords = torch.tensor([bb_coords_label, bb_coords_output])
    print(bb_coords)
    im_with_bb = torchvision.utils.draw_bounding_boxes(image=rand_test_img_sample, boxes=bb_coords, colors=['white', 'blue'])
    plt.imshow(im_with_bb.permute(1, 2, 0))
    plt.show()
    


#im.show()