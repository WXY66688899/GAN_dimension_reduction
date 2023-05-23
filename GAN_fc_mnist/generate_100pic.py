# prerequisites
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
#导入celeba
#64*64

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
celeba_loader = torch.utils.data.DataLoader(dset, batch_size=bs, num_workers=4,shuffle=True)

#定义generator&generator

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 64)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.fc4(x)
        #x = torch.tanh(self.fc4(x))
        return x


class Generator_256(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator_256, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.fc4(x)
        #x = torch.tanh(self.fc4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

class Discriminator_1024(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator_1024, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

#导入CSNET
inputsize = dset[0][0].size()[1]
IMAGE_SIZE= dset[0][0].size()[1] * dset[0][0].size()[2]
MEASURE_SIZE= 588
net = CSNET_lite_3(IMAGE_SIZE, MEASURE_SIZE).to(device)
net.load_state_dict(torch.load('./CSNET_lite_3_MNIST_subrate_0.75/net_epoch_50_0.001240.pth'))
#model = CSNET_lite_3(IMAGE_SIZE, MEASURE_SIZE)
#model.load_state_dict(torch.load('./CSNET_lite_3_celeba_128_subrate_0.04998779296875/net_epoch_100_0.008197.pth'))

# build network开始训练gan
z_dim = 100#z的input

G = Generator_256(g_input_dim = z_dim, g_output_dim = IMAGE_SIZE).to(device)
G.load_state_dict(torch.load('./GAN_FC_mnist_28_zdim_100/generator_epoch_100_1.036212.pth'))
#refine_G = CSNET_lite_3(IMAGE_SIZE, MEASURE_SIZE).to(device)
#refine_G.load_state_dict(torch.load('./refine_CSGAN_FC_mnist_fashion_28_SUBRATE_0.75_zdim_100/refine_generator_epoch_50_7.383249.pth'))

un_norm = transforms.Normalize(
    mean=[-1],
    std=[1 / 0.5]
)
pic_number = 1000

save_path = './img_generate_1000_for_test/gan_mnist'
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i in range(pic_number):
    noise = Variable(torch.randn(1, z_dim).to(device))
    fake = G(noise).view(1,1,28,28)
    # fake = un_norm(net.fc3(net.fc2(G(noise))).view(showsize,1,28,28))
    #fake = refine_G(net.fc3(net.fc2(G(noise)))).view(1,1,28,28)
    save_image(fake, '%s/fake_samples_%s.png' % (save_path, i), nrow=8, normalize=True)



'''
save_path = './img_generate_1000_for_test/original_mnist'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#生成原图
for i in range(pic_number):
    img = dset[i][0]
    save_image(img, '%s/fake_samples_%s.png' % (save_path,i), nrow=1, normalize=True)
'''