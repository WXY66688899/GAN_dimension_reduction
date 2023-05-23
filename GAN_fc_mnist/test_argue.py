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


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data.dataset import ConcatDataset
    trainset = MNIST(
        root='MNIST/train/MNIST/raw', train=True, transform= transforms.ToTensor())
    trainset2 = MNIST(
        root='MNIST/train/MNIST/raw', train=True, warp=True, transform= transforms.ToTensor())
    x = ConcatDataset([trainset,trainset2])

    input_size = x[0][0].size()[1]
    IMAGE_SIZE = x[0][0].size()[1] * x[0][0].size()[2]

    img_show = x[5][0]
    save_image(img_show, './original' + '.png', nrow=1, padding=2, pad_value=255)
    img_affine = transform.RandomAffine(degrees=30, shear=80, fill=0)(img_show)
    img_affine = img_affine.view(1, input_size, input_size)
    save_image(img_affine, './affine' + '.png', nrow=1, padding=2, pad_value=255)

    #print(x[59999][1])
    #trainloader = torch.utils.data.DataLoader(x, batch_size=100, shuffle=True, num_workers=2)

    #img,target = next(iter(trainloader))
    #print(1)