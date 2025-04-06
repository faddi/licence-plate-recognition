import numpy as np
import matplotlib.pylab as plt
import matplotlib
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms.v2 as transforms
from glob import glob
import math
import random
import torchvision.transforms.functional as F
from torchvision import tv_tensors
from helper import plot

# set default device to mps

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def getRandomSampleImageAndLabel(path):
    image_path_list = glob(path + "images/*.png")
    image_path_list1 = image_path_list.sort()
    label_path_list = glob(path + "labels/*.txt")
    label_path_list1 = label_path_list.sort()
    lenth_image_list = len(image_path_list)
    rand_index = random.randint(0, lenth_image_list - 1)
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
    x, y = size
    boundingBoxMiddleInX = label[0] * x
    boundingBoxMiddleInY = label[1] * y
    width = label[2] * x
    height = label[3] * y
    return boundingBoxMiddleInX, boundingBoxMiddleInY, width, height


def plot_image_with_bb(image_tensor, bb_label_tensor, bb_output_tensor):
    bb_coords_label = getBoundingBoxCoords(bb_label_tensor, (128, 128))
    if bb_output_tensor is not None:
        bb_coords_output = getBoundingBoxCoords(bb_output_tensor, (128, 128))
        bb_coords = torch.tensor([bb_coords_label, bb_coords_output])
    else:
        bb_coords = torch.tensor([bb_coords_label])
    print(bb_coords)
    im_with_bb = torchvision.utils.draw_bounding_boxes(
        image=image_tensor, boxes=bb_coords, colors=["white", "blue"]
    )
    plt.imshow(im_with_bb.permute(1, 2, 0))
    plt.show()


data = np.loadtxt("Data/Train Data.csv", delimiter=",", dtype=np.float64, skiprows=1)

data_dir_sample = "Data/test/images/Cars2.png"

test = Image.open(data_dir_sample)
sample_img = plt.imread(data_dir_sample)

data_dir_train = "Data/train/"
data_dir_test = "Data/test/"

im, label = getRandomSampleImageAndLabel(data_dir_train)


boundingBoxCoords = getBoundingBoxCoords(label[1:], im.size)
# draw = ImageDraw.Draw(im)
# draw.rectangle(boundingBoxCoords)


class LicencePlateData(Dataset):
    def __init__(self, data_dir, transform):
        self.xData = glob(data_dir + "images/*.png")
        self.yData = glob(data_dir + "labels/*.txt")
        self.xData.sort()
        self.yData.sort()
        self.transform = transform
        self.n_samples = len(self.xData)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        feature_dir = self.xData[index]
        label_dir = self.yData[index]
        # print(feature_dir)
        # print(label_dir)
        # feature = torch.from_numpy(plt.imread(feature_dir)[:, :, :3])

        image = Image.open(feature_dir).convert("RGB")

        # feature = torch.from_numpy(plt.imread(feature_dir)[:, :, :3])
        label = torch.from_numpy(np.loadtxt(label_dir)[1:])

        label_converted = torchvision.ops.box_convert(
            label, in_fmt="cxcywh", out_fmt="xyxy"
        )

        (w, h) = image.size

        label_scaled = label_converted * torch.tensor([w, h, w, h])

        box = tv_tensors.BoundingBoxes(
            label_scaled.unsqueeze(0), format="xyxy", canvas_size=(h, w)
        )

        # feature = feature.permute(2, 0, 1)
        # image_transformed, label_transformed = self.transform((image, box))
        image_transformed, label_transformed = self.transform((image, box))

        # plot([(image_transformed, label_transformed)])

        label_untransformed = torchvision.ops.box_convert(
            label_transformed, in_fmt="xyxy", out_fmt="cxcywh"
        )

        return image_transformed.float().to(device), label_untransformed.view(
            4
        ).float().to(device) / 128.0

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
            nn.MaxPool2d(2, 2),
            # output: 1024 x 4 x 4
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid(),
        )

    def forward(self, x, **kwargs):
        for i in self.network:
            x = i(x)

        return x
        # return self.network(x)


base_resize = 128


transform = transforms.Compose(
    [
        # transforms.Resize(base_resize),
        # transforms.CenterCrop(base_resize * 4),
        # transforms.Resize(base_resize),
        transforms.CenterCrop(base_resize * 4),
        transforms.Resize(base_resize),
        # transforms.RandomResizedCrop(base_resize),
        # transforms.ClampBoundingBoxes(),
        # transforms.SanitizeBoundingBoxes(labels_getter=lambda x: x[1]),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

training_data = LicencePlateData(data_dir_train, transform=transform)

test_data = LicencePlateData(data_dir_test, transform=transform)


# feature, label = training_data[66]
num_epochs = 40
batch_size = 40
learning_rate = 1e-4
nr_of_samples = training_data.n_samples
nr_of_iterations = math.ceil(batch_size / nr_of_samples)
nr_of_outputs = label.shape[0]

training_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


model = NeuralNet()
model.to(device)
model.train()

criterion = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(training_loader)
print(n_total_steps)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(training_loader):
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        # loss = (outputs - labels).pow(2).abs().mean()

        loss.backward()
        optimizer.step()
        mean_grad = model.network[0].weight.grad.mean()
        print(
            f"epoch {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item()} mean grad= {mean_grad}"
        )

# test
with torch.no_grad():
    test_loss = 0
    for test_images, test_labels in training_loader:
        # for test_images, test_labels in test_loader:
        test_outputs = model(test_images)
        # test_loss += criterion(test_outputs, test_labels)
        rand_test_img_sample = test_images[0]
        rand_test_img_label = test_labels[0]
        rand_test_output = test_outputs[0]

        label_transformed = torchvision.ops.box_convert(
            rand_test_img_label, in_fmt="cxcywh", out_fmt="xyxy"
        )

        label_transformed = label_transformed * 128

        output_transformed = torchvision.ops.box_convert(
            rand_test_output, in_fmt="cxcywh", out_fmt="xyxy"
        )

        output_transformed = output_transformed * 128

        label_bb = tv_tensors.BoundingBoxes(
            label_transformed.view(1, 4), format="xyxy", canvas_size=(128, 128)
        )

        output_bb = tv_tensors.BoundingBoxes(
            output_transformed.view(1, 4), format="xyxy", canvas_size=(128, 128)
        )

        plot(
            [
                (rand_test_img_sample, label_bb),
                (rand_test_img_sample, output_bb),
            ]
        )

        # bb_coords_label = getBoundingBoxCoords(rand_test_img_label, (128, 128))
        # bb_coords_output = getBoundingBoxCoords(rand_test_output, (128, 128))
        # bb_coords = torch.tensor([bb_coords_label, bb_coords_output])
        # print(bb_coords)
        # im_with_bb = torchvision.utils.draw_bounding_boxes(
        #     image=rand_test_img_sample, boxes=bb_coords, colors=["white", "blue"]
        # )
        # plt.imshow(im_with_bb.permute(1, 2, 0))
        plt.show()


# im.show()
