import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
from kornia.geometry.transform import translate,rotate
import torch.nn.functional as F
import kornia.morphology as morph


def expand_true_neighbors(mask):
    mask = mask.float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)  # 3x3 kernel

    # Perform 2D convolution (binary dilation)
    dilated = F.conv2d(mask, kernel, padding=1)
    expanded = (dilated > 0).squeeze(0).squeeze(0)  # Any neighboring True makes it True

    return expanded.bool()

def slivermask(img: torch.Tensor, sliver_width: int):

    # build a square kernel of side = 2*sliver_width+1
    kernel = torch.ones((2*sliver_width+1, 2*sliver_width+1),
                        dtype=img.dtype, device=img.device)
    # first erode, then dilate
    opened = morph.opening(img, kernel)

    return opened>=torch.mean(opened)

def remove_slivers(img: torch.Tensor):
    mask=slivermask(img,3)
    im=torch.zeros_like(img)
    im[mask]=img[mask]
    return im

def center_and_orient(im):
    h,w=im.shape
    x=torch.linspace(-w//2,w//2,w)
    y=torch.linspace(-h//2,h//2,h)
    y,x=torch.meshgrid(x,y)

    mass=torch.sum(im)

    x_cent=torch.sum(im*x)/mass
    y_cent=torch.sum(im*y)/mass

    shift=-torch.tensor([x_cent,y_cent])

    shifted=translate(im[None,None,...],shift[None,:]).squeeze()

    cov=torch.tensor([
        [torch.sum(x**2*shifted),torch.sum(x*y*shifted)],
        [torch.sum(x*y * shifted), torch.sum( y**2 * shifted)]
    ])
    vals,vecs=torch.linalg.eigh(cov)
    vec=vecs[-1]

    theta=torch.atan2(vec[1],vec[0])

    rotated=rotate(shifted[None,None,...],theta[None]*(180/torch.pi)).squeeze()

    #fig,ax=plt.subplots(1,2)
    #ax[0].imshow(im, cmap="gray")
    #ax[1].imshow(rotated,cmap="gray")
    #plt.show()
    return rotated




dir="VAE_single_cell2"
writedir="VAE_single_cell2_rotated"

n1=6
n2=8
nimages=n1*n2

images=torch.zeros((nimages,64,64))
adjusted=torch.zeros((nimages,64,64))

num_files = sum(os.path.isfile(os.path.join(dir, f)) for f in os.listdir(dir))

for i,filename in enumerate(os.listdir(dir)):
    print(f"\r{i*100/num_files}%",end="")

    count = i % nimages
    filepath=os.path.join(dir,filename)
    ar=tifffile.imread(filepath)
    tens=torch.from_numpy(ar)

    adjusted=center_and_orient(tens)
    tifffile.imwrite(os.path.join(writedir,filename),np.array(adjusted).astype(np.float32))

    '''
    images[i%nimages]=torch.from_numpy(ar)
    if (i+1)%nimages==0:
        for j in range(nimages):
            #adjusted[j]=remove_slivers(images[j][None,None,...]).squeeze()
            adjusted[j]=center_and_orient(images[j])

    if (i+1)%nimages==0:
        print("here")
        fig1,ax1=plt.subplots(n1,n2)
        for j,ax in enumerate(ax1.flatten()):
            ax.imshow(images[j],cmap="gray")

        fig2, ax2 = plt.subplots(n1, n2)
        for j, ax in enumerate(ax2.flatten()):
            ax.imshow(adjusted[j], cmap="gray")

        plt.show()
    '''








