import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy
import tifffile
import os

dir="SingleParticleImages/interphase-control"
'''
for filename in os.listdir(dir):
    filepath=os.path.join(dir,filename)
    if os.path.isfile(filepath):
        tens=torch.load(filepath)
        activation=((tens.max()-tens.min())*0.1+tens.min()).item()
        arr=np.array(tens,dtype=float)

        xarr=np.all(arr<activation,axis=1)
        yarr=np.all(arr<activation,axis=0)

        xdiffs=np.diff(xarr.astype(int))
        ydiffs=np.diff(yarr.astype(int))

        x1=np.where(xdiffs==-1)[0][0]
        x2=np.where(xdiffs==1)[0][0]+2

        y1=np.where(ydiffs==-1)[0][0]
        y2=np.where(ydiffs==1)[0][0]+2

        x=[x1,x2,x2,x1,x1]
        y=[y1,y1,y2,y2,y1]

        print(x)
        print(y)

        plt.imshow(arr)

        plt.plot(y,x,'r-')
        plt.show()
'''



def pad_tensor(tensor, cent, newshape):
    # newshape: (H, W)
    # cent: (x, y) in tensor coordinates
    # tensor: shape (h, w)

    bgsamp = [33050, 33058, 33058, 33062, 33072, 33066, 33062, 33072, 33066, 33066,
              33056, 33065, 33072, 33072, 33068, 33072, 33081, 33090]
    sig = np.std(np.array(bgsamp))
    mu = np.mean(np.array(bgsamp))

    tgt_cent = torch.tensor([newshape[0] // 2, newshape[1] // 2])  # (y, x)
    dx = tgt_cent - torch.flip(cent, [0])  # how much to shift original tensor

    # Create new image with noise
    new_im = (torch.randn(newshape) * sig + mu).to(torch.int)

    h, w = tensor.shape
    y0 = dx[0].item()
    x0 = dx[1].item()

    # Coordinates in new image
    y_start = max(0, y0)
    x_start = max(0, x0)
    y_end = min(newshape[0], y0 + h)
    x_end = min(newshape[1], x0 + w)

    # Coordinates in old tensor
    y_src_start = max(0, -y0)
    x_src_start = max(0, -x0)
    y_src_end = y_src_start + (y_end - y_start)
    x_src_end = x_src_start + (x_end - x_start)

    # Paste old tensor into new one
    new_im[y_start:y_end, x_start:x_end] = tensor[y_src_start:y_src_end, x_src_start:x_src_end]

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


l=150

shapes=[]

for filename in os.listdir(dir):
    filepath=os.path.join(dir,filename)
    if os.path.isfile(filepath):
        tens=torch.load(filepath)

        tensnorm=tens.to(torch.float)
        tensnorm=(tensnorm-tensnorm.min())/(tensnorm.max()-tensnorm.min())

        w=tens.shape[0]
        h=tens.shape[1]
        xadd=150-w
        yadd=150-h

        cent=mass_centroid(tensnorm.view((1,1,w,h))).squeeze()

        cent=cent.to(torch.int)
        cent+=torch.tensor([h//2,w//2])

        newim=pad_tensor(tens,cent,(l,l))

        fig,axs=plt.subplots(1,2)

        axs[0].imshow(tens,cmap="gray")
        axs[0].scatter(cent[0],cent[1])

        axs[1].imshow(newim,cmap="gray")
        axs[1].scatter([l//2],[l//2])
        newimnorm = newim.to(torch.float)
        newimnorm = (newimnorm-newimnorm.min())/(newimnorm.max()-newimnorm.min())
        print(mass_centroid(newimnorm.view(1,1,l,l)))

        plt.show()
        #plt.imshow(tensnorm,cmap="gray")
        #plt.scatter(cent[0],cent[1])
        #plt.show()


