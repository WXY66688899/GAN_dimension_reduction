from __future__ import print_function
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from csnet import CSNet_Lite, _netG2, _netRS, CSNet
from refiner_model import _RefinerG_64, _RefinerD_64, UnetGenerator
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from aegan_model import _netG1, _netD1, _netD2,reshape,recover, AEGAN_ResnetGenerator, AEGAN_ResnetDecoder, \
    NLayerDiscriminator
from csnet import CSNet_Lite, _netG2, _netRS, CSNet
from refiner_model import _RefinerG_64, _RefinerD_64, UnetGenerator


from tqdm import tqdm
from data_utils import TrainDatasetFromFolder, TestDatasetFromFolder
# from calculate_fid_pytorch.fid import fid_score
import vutils

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nm', type=int, default=63)

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--lrRS', type=float, default=0.001, help='learning rate, default=0.00001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--netG2', default='', help="path to netG2 (to continue training)")
parser.add_argument('--netRS', default='', help="path to netRS (to continue training)")
parser.add_argument('--RefinerG', default='', help="path to Refiner (to continue training)")
parser.add_argument('--RefinerD', default='', help="path to RefinerD (to continue training)")
parser.add_argument('--outf', default='generate_pic', help='folder to output images and model checkpoints')
parser.add_argument('--nc', type=int, default=3, help='the number of image channel')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')

parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--train_stage', type=int, default=1, help='training stage index')
parser.add_argument('--netG1', default='', help="path to netG1 (to continue training)")
parser.add_argument('--netD1', default='', help="path to netD1 (to continue training)")
parser.add_argument('--niter_stage2', type=int, default=200, help='number of epochs to train for G1 and D1')
parser.add_argument('--sub_dir2', default='imgStep2&3_2_csnet_lite_subrate_0.191_gan_fc_nor', help='the sub directory 2 of saving images')
parser.add_argument('--sub_dir3', default='generate_unnorm_194', help='the sub directory 3 of saving images')
parser.add_argument('--gan_type', type=str, default='GAN', help='options: GAN | Patch | PatchLSGAN')

parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--sub_rate', default=0.25, type=float, help='sampling sub rate')
parser.add_argument('--generatorWeights', type=str, default='', help="path to CSNet weights (to continue training)")

opt = parser.parse_args()
CROP_SIZE = opt.crop_size
BLOCK_SIZE = opt.block_size
SUBRATE = opt.sub_rate
LOAD_EPOCH = 0

# set gpu ids
str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
pic_number = 3000

#图片变形
RESHAPE = reshape(reshape_size=32)
RECOVER = recover(recover_size=2)


#导入压缩感知网络RS G2
# design model
#32*32*64生成512*512的图片
netG2 = _netG2(blocksize=BLOCK_SIZE, subrate=SUBRATE)
netG2.load_state_dict(torch.load('./CSNet_lite_normalize_patchsize_96_3tuple_subrate_0.25_blocksize_32/netG2_final_0.000493.pth'))
#512*512压缩为32*32*64
netRS = _netRS(blocksize=BLOCK_SIZE, subrate=SUBRATE)
netRS.load_state_dict(torch.load('./CSNet_lite_normalize_patchsize_96_3tuple_subrate_0.25_blocksize_32/netRS_final_0.000493.pth'))

#导入gan
#将noise生成为32*32*64
netG1 = _netG1(nz, ngf, opt.batchSize)
# netG1 = AEGAN_ResnetDecoder(nz, 64, n_blocks=1, n_downsampling=2)
# netG1.apply(weights_init)
netG1.load_state_dict(torch.load('GAN_nor_auto refine_STEP2&3_64_3tuple_subrate_0.25_blocksize_32/netG1_epoch_194_1.288953.pth'))

#自编码去噪器
RefinerG = _RefinerG_64(nc, ngf)
RefinerG.load_state_dict(torch.load('Refine_nor_auto refine_STEP2&3_64_3tuple_subrate_0.25_blocksize_32/RefinerG_epoch_194_138.292053.pth'))



netG1.cuda()
netG2.cuda()
netRS.cuda()
RefinerG.cuda()


un_norm = transforms.Normalize(
    mean=[-1],
    std=[1 / 0.5]
)


save_path = os.path.join(opt.outf, opt.sub_dir3)
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i in range(pic_number):
    noise = torch.randn(1, nz, 1, 1).cuda()
    #img_unrefine = netG2(RECOVER(netG1(noise).view(1, 3, 28, 28)))
    img = un_norm(RefinerG(netG2(RECOVER(netG1(noise)))))
    #vutils.save_image(img_unrefine, '%s/%s/fake_samples_unrefine_epoch_52_2.png' % (opt.outf, opt.sub_dir3), nrow=1, normalize=True)
    vutils.save_image(img, '%s/%s/fake_sample_%s.png' % (opt.outf, opt.sub_dir3,int(i)), nrow=1, normalize=True)

