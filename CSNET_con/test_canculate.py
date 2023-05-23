# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
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
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from csnet import CSNet_Lite, _netG2, _netRS,_netG2_block

# Device configuration,在cpu或cuda上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


bs = 100#batchsize
#导入celeba
'''
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
'''
#64*64
data_dir_64 = 'E:\CSGAN_fc\data\celeba-64'          # this path depends on your computer
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
'''
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
'''
dset = dset_64
dset_test = dset_64

inputsize = dset[0][0].size()[1]
IMAGE_SIZE= dset[0][0].size()[1] * dset[0][0].size()[2]
SUBRATE = 0.75
MEASURE_SIZE= int(IMAGE_SIZE*SUBRATE)
BLOCK_SIZE = 32



netG2 = _netG2_block(blocksize=BLOCK_SIZE, subrate=SUBRATE)
netG2.load_state_dict(torch.load('CSNet_normalize_patchsize_96_3tuple_subrate_0.75_blocksize_32_cropsize_96/netG2_final_0.006410.pth'))
#512*512压缩为32*32*64
netRS = _netRS(blocksize=BLOCK_SIZE, subrate=SUBRATE)
netRS.load_state_dict(torch.load('CSNet_normalize_patchsize_96_3tuple_subrate_0.75_blocksize_32_cropsize_96/netRS_final_0.006410.pth'))



un_norm = transforms.Normalize(
    mean=[-1],
    std=[1 / 0.5]
)


if torch.cuda.is_available():
    netG2.cuda()
    netRS.cuda()

test_img_real = dset_test[0][0].cuda()

save_img = 'celeba_64_csnet_contrast'
if not os.path.exists(save_img):
    os.makedirs(save_img)
save_image(test_img_real, save_img + '/real_samples.png' , nrow=1)

fake = un_norm(netG2(netRS(test_img_real.view(1,3,inputsize,inputsize))))
save_image(fake.data, save_img + '/fake_samples_%6f.png' % (SUBRATE), nrow=1)



##批量导入图片计算平均值
epoch_num = 1
psnr_final = np.zeros(epoch_num)
ssim_final = np.zeros(epoch_num)
for p in range(epoch_num):
    N = 32
    seed = np.random.randint(1, 10000, size=N)
    psnr_sum = np.zeros(N)
    ssim_sum = np.zeros(N)
    j = 0
    for i in seed:
        immg = dset[i][0].reshape(1, -1)
        img = dset[i][0].reshape(inputsize, inputsize, 3)
        img = un_norm(img).numpy()
        # 进入CSNET查看恢复情况
        immg = immg.cuda()
        immg_recon = netG2(netRS(immg.view(1, 3, inputsize, inputsize))).reshape(inputsize, inputsize, 3)
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
