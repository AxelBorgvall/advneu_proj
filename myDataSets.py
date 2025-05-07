import tifffile
import torch
import numpy as np
from torch.utils.data import Dataset
import os

def pad_tensor(tensor, cent, newshape):
    # newshape: (H, W)
    # cent: (x, y) in tensor coordinates
    # tensor: shape (h, w)

    HIGHEST_VALUE = 36863.0
    LOWEST_VALUE = 32995.0

    bgsamp = [33050, 33058, 33058, 33062, 33072, 33066, 33062, 33072, 33066, 33066,
              33056, 33065, 33072, 33072, 33068, 33072, 33081, 33090]
    sig = np.std(np.array(bgsamp))
    mu = np.mean(np.array(bgsamp))

    new_im = (torch.randn(newshape) * sig + mu)
    new_im = (new_im - LOWEST_VALUE) / (HIGHEST_VALUE - LOWEST_VALUE)

    new_im[new_im<0]=0

    tgt_cent = torch.tensor([newshape[0] // 2, newshape[1] // 2])  # (y, x)

    cent=torch.flip(cent.squeeze(),[0])

    cent=(cent+torch.tensor(tensor.shape)/2).to(torch.int)

    dx = tgt_cent - cent  # how much to shift original tensor

    new_im[dx[0]:min(dx[0]+tensor.shape[0],newshape[0]),dx[1]:min(dx[1]+tensor.shape[1],newshape[1])]=tensor

    return new_im




def mass_centroid(tensor):
    # tensor: (B, C, H, W)
    B, C, H, W = tensor.shape

    device = tensor.device

    y_coords = torch.linspace(-H/2,H/2,steps=H, device=device).float()
    x_coords = torch.linspace(-W/2,W/2,steps=W, device=device).float()
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)


    x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    mass = tensor.sum(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
    mass = mass + 1e-8

    x_centroid = (tensor * x_grid).sum(dim=(-2, -1), keepdim=False) / mass.squeeze(-1).squeeze(-1)
    y_centroid = (tensor * y_grid).sum(dim=(-2, -1), keepdim=False) / mass.squeeze(-1).squeeze(-1)

    centroids = torch.stack((x_centroid, y_centroid), dim=-1)
    return centroids


class SingleCellDataset(Dataset):
    HIGHEST_VALUE=36863.0
    LOWEST_VALUE=32995.0
    def __init__(self, dir, target_l,repeat=1):
        # Assuming inputs and targets are lists, convert them to tensors
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.inputs = torch.stack([torch.tensor(i) for i in inputs] * repeat)
        #self.inputs=self.inputs.to(self.device)
        images=[]
        #Loop over images in target directory
        for filename in os.listdir(dir):
            filepath = os.path.join(dir, filename)
            if os.path.isfile(filepath):
                images.append(torch.load(filepath))

        for i,im in enumerate(images):
            # normalize according to max and min in dataset
            im=im.to(torch.float)
            im=(im-self.LOWEST_VALUE)/(self.HIGHEST_VALUE-self.LOWEST_VALUE)
            # Pad to uniform size with plenty of space to transform
            cent=mass_centroid(im[None,None,...])
            im=pad_tensor(im,cent,(target_l,target_l)).view(1,target_l,target_l)
            images[i]=im

        # arrange into tensor and move to CUDA
        self.inputs = torch.stack([torch.tensor(i) for i in images] * repeat)

    def __len__(self):
        return self.inputs.__len__()

    def __getitem__(self, idx):
        return self.inputs[idx]



class VaeDataset(Dataset):
    def __init__(self, image_dir, angles_path, transform=None):
        self.paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith('.tif')
        ]
        self.angles = np.load(angles_path).astype('float32')  # Ensure float32
        assert len(self.paths) == len(self.angles)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = tifffile.imread(self.paths[idx]).astype('float32')
        img = torch.from_numpy(img).unsqueeze(0)  # shape (1, H, W)

        angle = torch.tensor([self.angles[idx]])  # shape (1,)
        if self.transform:
            img = self.transform(img)

        return img, angle


