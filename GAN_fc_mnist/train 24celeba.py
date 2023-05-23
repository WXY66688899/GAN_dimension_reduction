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
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
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
IMGSIZE = dset[0][0].size()[1]
IMAGE_SIZE= dset[0][0].size()[1] * dset[0][0].size()[2]
MEASURE_SIZE= IMAGE_SIZE
#model = CSNET_lite_3(IMAGE_SIZE, MEASURE_SIZE)
#model.load_state_dict(torch.load('./CSNET_lite_3_celeba_128_subrate_0.04998779296875/net_epoch_100_0.008197.pth'))

# build network开始训练gan
z_dim = 100#z的input

G = Generator(g_input_dim = z_dim, g_output_dim = MEASURE_SIZE).to(device)
D = Discriminator(MEASURE_SIZE).to(device)

# loss
criterion = nn.BCELoss()


if __name__=="__main__":
    # optimizer
    lr_G = 0.0001
    lr_D = 0.0001
    G_optimizer = optim.Adam(G.parameters(), lr=lr_G)
    D_optimizer = optim.Adam(D.parameters(), lr=lr_D)#将D优化器变为了SGD


    def D_train(x):
        # =======================Train the discriminator=======================#
        D.zero_grad()

        # train discriminator on real
        x_real, y_real = x.view(-1, MEASURE_SIZE), torch.ones(batch_size, 1)
        x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

        D_output = D(x_real)
        D_real_loss = criterion(D_output, y_real)
        D_real_score = D_output

        # train discriminator on facke
        z = Variable(torch.randn(bs, z_dim).to(device))
        x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

        D_output = D(x_fake)
        D_fake_loss = criterion(D_output, y_fake)
        D_fake_score = D_output

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()

        return D_loss.data.item()


    def G_train(x):
        # =======================Train the generator=======================#
        G.zero_grad()

        z = Variable(torch.randn(bs, z_dim).to(device))
        y = Variable(torch.ones(bs, 1).to(device))

        G_output = G(z)
        D_output = D(G_output)
        G_loss = criterion(D_output, y)

        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        G_optimizer.step()

        return G_loss.data.item()


    n_epoch = 150
    for epoch in range(1, n_epoch + 1):
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(celeba_loader):
            batch_size = x.size()[0]
            x = x.view(batch_size, IMGSIZE*IMGSIZE)
            x1=x
            #x1 = model.fc1(x)
            D_losses.append(D_train(x1))
            G_losses.append(G_train(x1))

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        img_dir = './img_mnist'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if epoch % 5 == 0:
            showsize = 16
            noise = Variable(torch.randn(showsize, z_dim).to(device))
            # vutils.save_image(real_cpu,'%s/%s/real_samples.png' % (opt.outf, opt.sub_dir2), normalize=True)
            fake = G(noise).view(showsize,1,28,28)
            save_image(fake, '%s/fake_samples_epoch_%s.png' % (img_dir, epoch),nrow=4, normalize=True)

        # for saving model
        save_dir = './GAN_FC_mnist_28' + '_zdim_' + str(z_dim)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if epoch % 5 == 0:
            torch.save(G.state_dict(), save_dir + '/generator_epoch_%d_%6f.pth' % (
                epoch, torch.mean(torch.FloatTensor(G_losses))))
            torch.save(D.state_dict(), save_dir + '/discriminator_epoch_%d_%6f.pth' % (
                epoch, torch.mean(torch.FloatTensor(D_losses))))

