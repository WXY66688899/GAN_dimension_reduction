# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from network import CSNET_sda
from network import CSNET_lite_3
from network import CSNET_fc_3
from network import CSNET_lite_5
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
from network import CSNET_lite_3


# Device configuration,在cpu或cuda上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 100#batchsize

#导入数据集

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


dset = dataset_mnist
#print(dset[0][0].size()[0])
celeba_loader = torch.utils.data.DataLoader(dset, num_workers=4, batch_size=bs, shuffle=True)
#net defination
inputsize = dset[0][0].size()[1]
IMAGE_SIZE= dset[0][0].size()[1] * dset[0][0].size()[2]
MEASURE_SIZE= 196
net = CSNET_lite_3(IMAGE_SIZE, MEASURE_SIZE)
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
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))
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

        # 图片重构对比
        test_img_real = dset[1][0].cuda()
        if epoch == 0:
            save_image(test_img_real, './img1_mnist_0.25_contrast/real_samples.png' , nrow=1)

        if epoch % 5 == 0:
            fake = un_norm(net(test_img_real.view(1,784)).view(1,1,28,28))
            save_image(fake.data, './img1_mnist_0.25_contrast/fake_samples_epoch_%s.png' % (int(epoch)), nrow=1)

        # for saving model
        save_dir = 'CSNET_lite_3_MNISTdgist' + '_subrate_' + str(MEASURE_SIZE/IMAGE_SIZE)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if epoch % 2 == 0:
            torch.save(net.state_dict(), save_dir + '/net_epoch_%d_%6f.pth' % (
                epoch, running_results['loss'] / running_results['batch_sizes']))
