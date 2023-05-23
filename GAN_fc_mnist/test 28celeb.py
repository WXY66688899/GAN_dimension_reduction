
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF
import torch
import matplotlib.pyplot as plt
import time, math, glob
from torchvision import datasets, transforms
# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from network import CSNET_lite_3
import os
from network import CSNET_FC_2

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

# Device configuration,在cpu或cuda上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#导入generator

z_dim = 100#z的input
MEASURE_SIZE= 28*28
G = Generator(g_input_dim = z_dim, g_output_dim = MEASURE_SIZE).to(device)
G.load_state_dict(torch.load('./gan_celeba28_lr_adam+adam0.0001_zdim_100/generator_epoch_65_1.496941.pth'))

bs = 25
with torch.no_grad():
    test_z = Variable(torch.randn(bs, z_dim).to(device))
    print(test_z.size())
    print(G(test_z).size())
    generated = G(test_z)
    #generated = model.fc3(G(test_z))
    print(generated.size())

    un_norm = transforms.Normalize(
        mean=[-1],
        std=[1 / 0.5]
    )

    generated = generated.view(generated.size(0), 1, 28, 28)
    #img = generated
    img = un_norm(generated)
    #img = generated.view(generated.size(0), 1, 128, 128)
    save_dir = 'generate_img'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_image(img, './generate_img/celeba_28adam+adam_0.0001_epoch_65' + '.png', nrow=5, padding=2, pad_value=255)  # nrow每行的张数，padding相邻图像间隔，pad_value图