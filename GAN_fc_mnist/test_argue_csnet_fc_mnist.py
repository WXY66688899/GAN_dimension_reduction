
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF
import torch
import matplotlib.pyplot as plt
import time, math, glob
import numpy as np
import random
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import matplotlib.pyplot as plt
import time, math, glob
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.color import rgb2ycbcr

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from network import CSNET_lite_3
from network import CSNET_sda
import argparse
from tqdm import tqdm
import os
from network import CSNET_fc_3


# Device configuration,在cpu或cuda上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 100#batchsize

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
celeba_loader = torch.utils.data.DataLoader(dset, num_workers=0, batch_size=bs, shuffle=True)
#net defination
input_size = dset[0][0].size()[1]
IMAGE_SIZE = dset[0][0].size()[1] * dset[0][0].size()[2]
subrate = 0.5
MEASURE_SIZE = int(IMAGE_SIZE * subrate)

#导入模型
net = CSNET_lite_3(IMAGE_SIZE, MEASURE_SIZE)
net.load_state_dict(torch.load('./CSNET_fc3_lite_mnist_subrate_0.5lr_0.00005_alpha_0.001_degrees_80_shear_150/net_epoch_50_0.002138.pth'))

#去归一化
un_norm = transforms.Normalize(
        mean=[-1],
        std=[1 / 0.5]
    )

##批量导入图片计算平均值
epoch_num = 1
psnr_final = np.zeros(epoch_num)
ssim_final = np.zeros(epoch_num)
for p in range(epoch_num):
    N = 32
    seed = np.random.randint(1, 30000, size=N)
    psnr_sum = np.zeros(N)
    ssim_sum = np.zeros(N)
    j = 0
    for i in seed:
        immg = dset[i][0].reshape(1,-1)
        img = dset[i][0].reshape(input_size, input_size, 1)
        img = un_norm(img).numpy()
        # 进入CSNET查看恢复情况
        immg = immg
        immg_recon = net(immg)[0]
        immg_recon = immg_recon.reshape(input_size, input_size, 1)
        img_recon = un_norm(immg_recon)
        img_recon = img_recon.cpu().detach().numpy()


        # psnr计算
        psnr = compare_psnr(img, img_recon, data_range=1)
        psnr_sum[j] = psnr

        # ssim计算
        ssim = compare_ssim(img, img_recon, gaussian_weights=True, multichannel=True, data_range=1)
        ssim_sum[j] = ssim
        j = j + 1

    psnr_ave = np.sum(psnr_sum) / N
    psnr_final[p] = psnr_ave
    #print(psnr_ave)
    ssim_ave = np.sum(ssim_sum) / N
    ssim_final[p] = ssim_ave
    #print(ssim_ave)

psnr_last = np.sum(psnr_final) / epoch_num
print(psnr_last)
ssim_last = np.sum(ssim_final) / epoch_num
print(ssim_last)

save_dir = 'contrast_csnet_fc'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
img_show = dset[1][0]
save_image(img_show, './contrast_csnet_fc/original' + '.png', nrow=1, padding=2, pad_value=255)
img_reconshow = net(img_show.view(1, input_size*input_size))[0]
img_reconshow = img_reconshow.view(1, input_size, input_size)
save_image(img_reconshow, './contrast_csnet_fc/recon_csnet_lite_3_subrate_0.5_lr_0.0005_alpha_0.001_degree_80_shear_150' + '.png', nrow=1, padding=2, pad_value=255)
