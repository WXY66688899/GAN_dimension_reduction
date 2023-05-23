from lib.network import CSNet_Lite
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
from torchvision.utils import save_image
from data_utils import TestDatasetFromFolder
from lib.network import CSNet

input_size = 512
BLOCKSIZE = 8
#导入数据集1024
#1024*1024
data_dir_1024 = 'celeba_1024_lite/celeba_1024'          # this path depends on your computer
dset_1024 = TestDatasetFromFolder(data_dir_1024, blocksize=BLOCKSIZE)
#512*512
data_dir_512 = 'celeba_512_lite/celeba_512'          # this path depends on your computer
dset_512 = TestDatasetFromFolder(data_dir_512, blocksize=BLOCKSIZE)

#print(dset_1024[0])#第一张图两通道都一样
#print(dset_1024[0][0])
#print(dset_1024[0][1])#dset_1024[0][0]=dset_1024[0][1],torch.Size([1, 1024, 1024])

#immg = dset_1024[0][0]
#img = immg.numpy().reshape(input_size,input_size,1)
#print(max(img),min(img))0,1之间




#导入模型
model = CSNet_Lite(BLOCKSIZE, 0.01).cuda()#改变架构都要改
model.load_state_dict(torch.load('./CSNet_lite_patchsize_48_subrate_0.01_blocksize_8/net_epoch_20_0.007504.pth'))


##批量导入图片计算平均值
epoch_num = 1
psnr_final = np.zeros(epoch_num)
ssim_final = np.zeros(epoch_num)
for p in range(epoch_num):
    N = 32
    seed = np.random.randint(1, 1280, size=N)
    psnr_sum = np.zeros(N)
    ssim_sum = np.zeros(N)
    j = 0
    for i in seed:
        immg = dset_512[i][0].reshape(1,1,input_size, input_size)
        img = dset_512[i][0].numpy().reshape(input_size, input_size, 1)
        img_show = img.reshape(input_size, input_size)
        # 进入CSNET查看恢复情况

        immg = immg.cuda()
        immg_recon = model(immg)
        img_recon = immg_recon.cpu().detach().numpy().reshape(input_size, input_size, 1)

        img_recon_show = img_recon.reshape(input_size, input_size)

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


img_show = dset_512[1][0].cuda()
save_image(img_show, './contrast_csnet/original_512_1tuple_48' + '.png', nrow=1, padding=2, pad_value=255)
img_reconshow = model(img_show.view(1,1, input_size,input_size))
img_reconshow = img_reconshow.view(1, 1, input_size, input_size)
save_image(img_reconshow, './contrast_csnet/recon_512_1tuple_48_csnet_lite_subrate0.01' + '.png', nrow=1, padding=2, pad_value=255)
