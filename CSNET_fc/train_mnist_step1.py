# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from network import CSNET_sda
from network import CSNET_lite_3
from network import CSNET_fc_3
from network import CSNET_lite_5,CSNET_sda_lite
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.datasets as dsets
import os
from network import CSNET_lite_3,CSNET_sda
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Device configuration,在cpu或cuda上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 100#batchsize
#导入celeba
data_dir_original = 'F:/CSNET_LITE/data_lite/train'          # this path depends on your computer
dset_original = datasets.ImageFolder(data_dir_original, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]))
#64*64
data_dir_64 = 'F:/CSNET_LITE/celebA_hq/celeba_64'          # this path depends on your computer
dset_64 = datasets.ImageFolder(data_dir_64, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]))

#128*128
data_dir_128 = 'F:/CSNET_LITE/celebA_hq/celeba_128'          # this path depends on your computer
dset_128 = datasets.ImageFolder(data_dir_128, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]))

#256*256
data_dir_256 = 'F:/CSNET_LITE/celebA_hq/celeba_256'          # this path depends on your computer
dset_256 = datasets.ImageFolder(data_dir_256, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]))

#256*256
data_dir_512 = 'F:/CSNET_LITE/celebA_hq/celeba_512'          # this path depends on your computer
dset_512 = datasets.ImageFolder(data_dir_512, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]))


#256*256
data_dir_1024 = 'F:/CSNET_LITE/celebA_hq/celeba_1024'          # this path depends on your computer
dset_1024 = datasets.ImageFolder(data_dir_1024, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]))

dataset_mnist = torchvision.datasets.MNIST(root='./MNIST/train',train=True, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        #transforms.CenterCrop(28),
        #transforms.Resize(28),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]),download=True)

dataset_mnist_test = torchvision.datasets.MNIST(root='./MNIST/test',train=False, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        #transforms.CenterCrop(28),
        #transforms.Resize(28),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]),download=True)


dataset_mnist_fashion = torchvision.datasets.FashionMNIST(root='./MNIST-Fashion/train',train=True, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        #transforms.CenterCrop(28),
        #transforms.Resize(28),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]),download=True)

dataset_mnist_fashion_test = torchvision.datasets.FashionMNIST(root='./MNIST-Fashion/test',train=False, transform=transforms.Compose([
        #transforms.Scale(IMAGE_SIZE),
        #transforms.Grayscale(num_output_channels=1),#转为单通道灰度图片
        transforms.ToTensor(),
        #transforms.CenterCrop(28),
        #transforms.Resize(28),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ]),download=True)

dset = dataset_mnist_fashion
dset_test = dataset_mnist_fashion_test
#print(dset[0][0].size()[0])
celeba_loader = torch.utils.data.DataLoader(dset, num_workers=4, batch_size=bs, shuffle=True)
#net defination
inputsize = dset[0][0].size()[1]
IMAGE_SIZE= dset[0][0].size()[1] * dset[0][0].size()[2]
subrate = 0.75
MEASURE_SIZE= int(IMAGE_SIZE*subrate)
net = CSNET_fc_3(IMAGE_SIZE, MEASURE_SIZE)
#定义损失
mse_loss = nn.MSELoss()

un_norm = transforms.Normalize(
    mean=[-1],
    std=[1 / 0.5]
)


if torch.cuda.is_available():
    net.cuda()
    mse_loss.cuda()

if __name__=="__main__":
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.99))
    # 学习率衰减
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    # 训练网络
    LOAD_EPOCH = 0
    NUM_EPOCHS = 50

    # for step, (data, _) in enumerate(celeba_loader):
    # training

    # print("steop:{}, batch_x:{}, batch_y:{}".format(step, data.size(), _.size()))#steop:21, batch_x:torch.Size([100, 1, 64, 64]), batch_y:torch.Size([100])

    for epoch in range(LOAD_EPOCH, NUM_EPOCHS + 1):
        train_bar = tqdm(celeba_loader)
        running_results = {'batch_sizes': 0, 'loss': 0, }

        net.train()

        for data, target in train_bar:
            batch_size = data.size(0)
            if batch_size <= 0:
                continue

            running_results['batch_sizes'] += batch_size

            data = torch.reshape(data, (batch_size, -1))
            real_img = data.cuda()
            z = data.cuda()

            #if torch.cuda.is_available():
                #real_img = data.cuda()

            #if torch.cuda.is_available():
                #z = data.cuda()


            fake_img = net(z)
            optimizer.zero_grad()
            loss = mse_loss(fake_img, real_img)
            loss.backward()
            optimizer.step()

            running_results['loss'] += loss.item() * batch_size

            train_bar.set_description(desc='[%d] Loss: %.4f lr: %.7f' % (
                epoch, running_results['loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr']))
        scheduler.step()

        # for saving model
        save_dir = 'CSNET_fc3_fashionMNIST' + '_subrate_' + str(MEASURE_SIZE/IMAGE_SIZE)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if epoch % 2 == 0:
            torch.save(net.state_dict(), save_dir + '/net_epoch_%d_%6f.pth' % (
                epoch, running_results['loss'] / running_results['batch_sizes']))
