from torch import nn

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class CSNET_lite_3(nn.Module):
    def __init__(self, image_dim, measure_dim):
        super(CSNET_lite_3, self).__init__()
        self.fc1 = nn.Linear(image_dim, measure_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(measure_dim, measure_dim*2))
        self.fc3 = nn.Linear(measure_dim*2, image_dim)
        self.fc4 = nn.Sequential(
            nn.Linear(measure_dim, 1),
            nn.Sigmoid()
        )

    # forward method
    def forward(self, x):
        x = self.fc1(x)
        y = self.fc4(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x, y

class CSNET_lite_5(nn.Module):
    def __init__(self, image_dim, measure_dim):
        super(CSNET_lite_3, self).__init__()
        self.fc1 = nn.Linear(image_dim, measure_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(measure_dim, measure_dim*2))
        self.fc3 = nn.Sequential(
            nn.Linear(measure_dim*2, measure_dim *4))
        self.fc4 = nn.Sequential(
            nn.Linear(measure_dim*4, measure_dim * 8))
        self.fc5 = nn.Sequential(
            nn.Linear(measure_dim*8, measure_dim * 16))
        self.fc6 = nn.Linear(measure_dim*16, image_dim)

    # forward method
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x

class CSNET_fc_3(nn.Module):
    def __init__(self, image_dim, measure_dim):
        super(CSNET_fc_3, self).__init__()
        self.fc1 = nn.Linear(image_dim, measure_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(measure_dim, measure_dim*2),
            nn.Sigmoid())
        self.fc3 = nn.Sequential(
            nn.Linear(measure_dim*2, image_dim),
            nn.Sigmoid())

    # forward method
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class CSNET_sda(nn.Module):
    def __init__(self, image_dim, measure_dim):
        super(CSNET_sda, self).__init__()
        self.fc1 = nn.Linear(image_dim, measure_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(measure_dim, image_dim),
            nn.Sigmoid())
        self.fc3 = nn.Sequential(
            nn.Linear(image_dim, measure_dim),
            nn.Sigmoid())
        self.fc4 = nn.Sequential(
            nn.Linear(measure_dim, image_dim),
            nn.Sigmoid())
    # forward method
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class CSNET_sda_lite(nn.Module):
    def __init__(self, image_dim, measure_dim):
        super(CSNET_sda_lite, self).__init__()
        self.fc1 = nn.Linear(image_dim, measure_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(measure_dim, image_dim))
        self.fc3 = nn.Sequential(
            nn.Linear(image_dim, measure_dim))
        self.fc4 = nn.Sequential(
            nn.Linear(measure_dim, image_dim))
    # forward method
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class CSNET_FC_2(nn.Module):
    def __init__(self, image_dim, measure_dim):
        super(CSNET_FC_2, self).__init__()
        self.fc1 = nn.Linear(image_dim, measure_dim)
        #self.fc2 = nn.Sequential( nn.Linear(measure_dim, measure_dim*2))
        self.fc3 = nn.Linear(measure_dim, image_dim)


    # forward method
    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        x = self.fc3(x)
        return x


class _netRS(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1, channal=1):
        super(_netRS, self).__init__()
        self.blocksize = blocksize
        self.channal = channal

        self.sampling = nn.Conv2d(channal, int(np.round(blocksize * blocksize * subrate)) * channal, blocksize, stride=blocksize, padding=0, bias=False)

    def forward(self, x):
        x = self.sampling(x)
        return x

class _netG2(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1, channal=1):
        super(_netG2, self).__init__()
        self.blocksize = blocksize
        self.channal = channal
        # upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate))*channal, blocksize*blocksize*channal, 1, stride=1, padding=0)
    def forward(self, x):
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)  # Reshape + Concat
        return x


class reshape(nn.Module):
    def __init__(self, reshape_size):
        super(reshape, self).__init__()
        self.reshape_size = reshape_size
    def forward(self, input):
        bs = input.size()[0]
        nc = input.size()[1]
        measzie = input.size()[2]
        N_W, N_H = self.reshape_size // measzie, self.reshape_size // measzie
        N_T = N_H * N_W
        N_C = nc // N_T
        f = input.view(bs, nc // N_W, N_W, measzie, measzie).permute(0, 1, 3, 2, 4)
        z = f.contiguous().view(bs, N_C, self.reshape_size, self.reshape_size)
        return z

class recover(nn.Module):
    def __init__(self, recover_size):
        super(recover, self).__init__()
        self.recover_size = recover_size
    def forward(self, input):
        bs = input.size()[0]
        N_C = input.size()[1]
        measzie = input.size()[2]
        N_W, N_H = measzie // self.recover_size, measzie // self.recover_size
        N_T = N_H * N_W
        nc = N_C*N_T
        f = input.view(bs, N_C*N_W,self.recover_size,N_W, self.recover_size).permute(0, 1, 3, 2, 4)
        z = f.contiguous().view(bs, nc, self.recover_size, self.recover_size)
        return z


class refine_g(torch.nn.Module):
    def __init__(self):
        super(refine_g, self).__init__()
        self.l1 = torch.nn.Linear(784, 256)
        self.l2 = torch.nn.Linear(256, 64)
        self.l3 = torch.nn.Linear(64, 20)
        self.l4 = torch.nn.Linear(20, 64)
        self.l5 = torch.nn.Linear(64, 256)
        self.l6 = torch.nn.Linear(256, 784)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), 0.2)
        x = F.leaky_relu(self.l2(x), 0.2)
        x = F.leaky_relu(self.l3(x), 0.2)
        x = F.leaky_relu(self.l4(x), 0.2)
        x = F.leaky_relu(self.l5(x), 0.2)
        x = torch.sigmoid(self.l6(x))
        return x




'''
x = torch.randn(100, 3, 64, 64)
bs=x.size()[0]
image_dim = x.size()[2]*x.size()[3]
measure_dim = 784
x = torch.reshape(x,(bs,3,-1))
print(x.size())
net = CSNET_FC(image_dim,measure_dim)
z = net(x)
print(z.size())
meas = net.fc1(x)
print(meas.size())
'''