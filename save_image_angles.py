import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
from kornia.geometry.transform import translate,rotate
import kornia.morphology as morph

def get_angle(im):
    h, w = im.shape
    x = torch.linspace(-w // 2, w // 2, w)
    y = torch.linspace(-h // 2, h // 2, h)
    y, x = torch.meshgrid(x, y)

    mass = torch.sum(im)

    #x_cent = torch.sum(im * x) / mass
    #y_cent = torch.sum(im * y) / mass
    #shift = -torch.tensor([x_cent, y_cent])
    #shifted = translate(im[None, None, ...], shift[None, :]).squeeze()

    cov = torch.tensor([
        [torch.sum(x ** 2 * im), torch.sum(x * y * im)],
        [torch.sum(x * y * im), torch.sum(y ** 2 * im)]
    ])
    vals, vecs = torch.linalg.eigh(cov)
    vec = vecs[-1]

    theta = torch.atan2(vec[1], vec[0])*180/torch.pi

    return theta






dir="VAE_single_cell2"
writedir="VAE_single_cell2_rotated"

n1=6
n2=8
nimages=n1*n2

images=torch.zeros((nimages,64,64))
adjusted=torch.zeros((nimages,64,64))

num_files = sum(os.path.isfile(os.path.join(dir, f)) for f in os.listdir(dir))

angles=np.zeros(num_files)

for i,filename in enumerate(os.listdir(dir)):
    print(f"\r{i*100/num_files}%",end="")

    count = i % nimages
    filepath=os.path.join(dir,filename)
    ar=tifffile.imread(filepath)
    tens=torch.from_numpy(ar)

    angle=get_angle(tens)
    angles[i]=angle.item()
    '''
    rot=rotate(tens[None,None,...],angle[None])
    rotback=rotate(rot,-angle[None])
    fig,axs=plt.subplots(1,2)
    axs[0].imshow(tens,cmap="gray")
    axs[1].imshow(rotback.squeeze(),cmap="gray")
    plt.show()
    '''
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


np.save("VAE_single_cell2_angles",angles)
