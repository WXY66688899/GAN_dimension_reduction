# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from network import CSNET_sda
from network import CSNET_lite_3
from network import CSNET_fc_3
from network import CSNET_lite_5,CSNET_sda_lite
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
from network import CSNET_lite_3,CSNET_sda
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
from typing import Any, Callable, Optional, Tuple
from torchvision.utils import save_image
import torchvision.transforms as transform
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

class MNIST(data.Dataset):

    def __init__(
            self,
            root: str,
            train: bool = True,
            warp: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(MNIST, self).__init__()
        self.train = train  # training set or test set
        self.root = root
        self.transform = transform
        self.target_transform =target_transform
        self.data, self.targets = self._load_data()
        if warp:
            self.targets = torch.zeros_like(self.targets) + 1  # 扭曲标签为1
        else:
            self.targets = torch.zeros_like(self.targets)   # 否则标签为0

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.root, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.root, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}
def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2])).view(*s) ## torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

# Device configuration,在cpu或cuda上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 50#batchsize

#定义损失
mse_loss = nn.MSELoss()

un_norm = transforms.Normalize(
    mean=[-1],
    std=[1 / 0.5]
)



if __name__=="__main__":
    degree_size = 80
    shear_size = 150
    trainset = MNIST(root='MNIST/train/MNIST/raw', train=True, transform= transforms.ToTensor())
    trainset2 = MNIST(root='MNIST/train/MNIST/raw', train=True, warp=True, transform=transforms.Compose([
        transform.RandomAffine(degrees=degree_size, shear=shear_size, fill=0), transforms.ToTensor()]))
    dset = ConcatDataset([trainset, trainset2])
    dset_test = dset
    celeba_loader = torch.utils.data.DataLoader(dset, num_workers=0, batch_size=bs, shuffle=True)
    # net defination
    inputsize = dset[0][0].size()[1]
    IMAGE_SIZE = dset[0][0].size()[1] * dset[0][0].size()[2]
    subrate = 0.5
    MEASURE_SIZE = int(IMAGE_SIZE * subrate)
    net = CSNET_lite_3(IMAGE_SIZE, MEASURE_SIZE)

    if torch.cuda.is_available():
        net.cuda()
        mse_loss.cuda()
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.00005, betas=(0.9, 0.99))
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
        running_results = {'batch_sizes': 0, 'loss_recover': 0,'loss_lable': 0, }

        net.train()

        for data, target in train_bar:
            batch_size = data.size(0)
            if batch_size <= 0:
                continue

            running_results['batch_sizes'] += batch_size

            data = torch.reshape(data, (batch_size, -1))
            real_img = data
            z = data
            lable = target.float()

            #if torch.cuda.is_available():
                #real_img = data.cuda()

            #if torch.cuda.is_available():
                #z = data.cuda()


            fake_img = net(z)[0]
            fake_lable = net(z)[1]
            optimizer.zero_grad()
            loss1 = mse_loss(fake_img, real_img)
            loss1.backward()
            loss2 = mse_loss(fake_lable,lable)*0.001
            loss2.backward()
            optimizer.step()

            running_results['loss_recover'] += loss1.item() * batch_size
            running_results['loss_lable'] += loss2.item() * batch_size

            train_bar.set_description(desc='[%d] Loss1: %.4f Loss2: %.4f lr: %.7f' % (
                epoch, running_results['loss_recover'] / running_results['batch_sizes'], running_results['loss_lable'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr']))
        scheduler.step()

        # for saving model
        save_dir = 'CSNET_fc3_lite_mnist' + '_subrate_' + str(MEASURE_SIZE/IMAGE_SIZE) +'lr_0.00005_alpha_0.001_degrees_'+str(degree_size)+'_shear_'+str(shear_size)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if epoch % 2 == 0:
            torch.save(net.state_dict(), save_dir + '/net_epoch_%d_%6f.pth' % (
                epoch, running_results['loss_recover'] / running_results['batch_sizes']))
