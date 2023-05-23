from torch import nn

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


# Reshape + Concat layer

class Reshape_Concat_Adap(torch.autograd.Function):
    blocksize = 0

    def __init__(self, block_size):
        # super(Reshape_Concat_Adap, self).__init__()
        Reshape_Concat_Adap.blocksize = block_size

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)

        data = torch.clone(input_.data)
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        output = torch.zeros((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                              int(w_ * Reshape_Concat_Adap.blocksize), int(h_ * Reshape_Concat_Adap.blocksize))).cuda()

        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = data[:, :, i, j]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                # data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                                            Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize))
                # print data_temp.shape
                output[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize] += data_temp

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        input_ = torch.clone(inp.data)
        grad_input = torch.clone(grad_output.data)

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros((b_, c_, w_, h_)).cuda()
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = grad_input[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                            j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += torch.squeeze(data_temp)

        return Variable(output)


def My_Reshape_Adap(input, blocksize):
    return Reshape_Concat_Adap(blocksize).apply(input)


# The residualblock for reconstruction network
class ResidualBlock(nn.Module):
    def __init__(self, channels, has_BN = False):
        super(ResidualBlock, self).__init__()
        self.has_BN = has_BN
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if has_BN:
            self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if has_BN:
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        if self.has_BN:
            residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = x + residual
        residual = self.conv2(residual)
        if self.has_BN:
            residual = self.bn2(residual)
        residual = self.relu(residual)

        return residual

#  code of CSNet
class CSNet_Lite(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1):

        super(CSNet_Lite, self).__init__()
        self.blocksize = blocksize

        # for sampling
        self.sampling = nn.Conv2d(3, int(np.round(blocksize*blocksize*subrate))*3, blocksize, stride=blocksize, padding=0, bias=False)
        # upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate))*3, blocksize*blocksize*3, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.sampling(x)
        #x =  x + torch.randn(size=x.shape).cuda() * 0.49
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize) # Reshape + Concat
        return x

class _netRS(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1):
        super(_netRS, self).__init__()
        self.blocksize = blocksize

        self.sampling = nn.Conv2d(3, int(np.round(blocksize * blocksize * subrate)) * 3, blocksize, stride=blocksize, padding=0, bias=False)

    def forward(self, x):
        x = self.sampling(x)
        return x

class _netG2(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1):
        super(_netG2, self).__init__()
        self.blocksize = blocksize
        # upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate))*3, blocksize*blocksize*3, 1, stride=1, padding=0)
    def forward(self, x):
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)  # Reshape + Concat
        return x


class _netG2_block(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1):
        super(_netG2_block, self).__init__()
        self.blocksize = blocksize
        # upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate))*3, blocksize*blocksize*3, 1, stride=1, padding=0)
        # reconstruction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv5 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)  # Reshape + Concat

        block1 = self.conv1(x)
        block2 = self.conv2(block1)
        block3 = self.conv3(block2)
        block4 = self.conv4(block3)
        block5 = self.conv5(block4)
        return block5




#  code of CSNet
class CSNet(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1):

        super(CSNet, self).__init__()
        self.blocksize = blocksize

        # for sampling
        self.sampling = nn.Conv2d(3, int(np.round(blocksize*blocksize*subrate))*3, blocksize, stride=blocksize, padding=0, bias=False)
        # upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate))*3, blocksize*blocksize*3, 1, stride=1, padding=0)

        # reconstruction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv5 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.sampling(x)
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize) # Reshape + Concat

        block1 = self.conv1(x)
        block2 = self.conv2(block1)
        block3 = self.conv3(block2)
        block4 = self.conv4(block3)
        block5 = self.conv5(block4)

        return block5
    
    
#  code of CSNet_Plus (Enhanced version of CSNet)

class CSNet_Plus(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1):

        super(CSNet_Plus, self).__init__()
        self.blocksize = blocksize

        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)

        # reconstruction network
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block2 = ResidualBlock(64, has_BN = False)
        self.block3 = ResidualBlock(64, has_BN = False)
        self.block4 = ResidualBlock(64, has_BN = False)
        self.block5 = ResidualBlock(64, has_BN = False)
        self.block6 = ResidualBlock(64, has_BN = False)

        self.block7 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.sampling(x)
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize) # Reshape + Concat

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)

        return x, block7 + x

if __name__ == '__main__':
    import torch

    img = torch.randn(1, 1, 32, 32).cuda()
    net = CSNet_Plus().cuda()
    out = net(img)
    print(net)
    print(out[1].size())

