import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
from kornia.geometry.transform import translate,rotate
import torch.nn.functional as F
import kornia.morphology as morph





dir= "../data/VAE_single_cell2_rotated"
writedir="../data/VAE_single_cell2_rotated"

n1=6
n2=8
nimages=n1*n2

images=torch.zeros((nimages,64,64))
adjusted=torch.zeros((nimages,64,64))

num_files = sum(os.path.isfile(os.path.join(dir, f)) for f in os.listdir(dir))

for i,filename in enumerate(os.listdir(dir)):
    print(f"\r{i*100/num_files}%",end="")

    refnoise=0.02

    count = i % nimages
    filepath=os.path.join(dir,filename)
    ar=tifffile.imread(filepath)
    tens=torch.from_numpy(ar)

    noiselevel=(refnoise*2)/2

    noise=torch.randn((64,64))*0.1*noiselevel+noiselevel
    mask=tens<=0.015
    tens[mask]=noise[mask]

    tifffile.imwrite(os.path.join(writedir,filename),np.array(tens))

    '''
    images[i%nimages]=tens
    
    if (i+1)%nimages==0:
        print("here")
        fig1,ax1=plt.subplots(n1,n2)
        for j,ax in enumerate(ax1.flatten()):
            ax.imshow(images[j],cmap="gray")

        #fig2, ax2 = plt.subplots(n1, n2)
        #for j, ax in enumerate(ax2.flatten()):
        #    ax.imshow(adjusted[j], cmap="gray")

        plt.show()
    '''



