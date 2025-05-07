import os
import sys
import tifffile
import torch
import matplotlib.pyplot as plt


nims=8*12
dir="VAE_single_cell2"

ims=torch.zeros((nims,64,64))

maxfound=0

for filenum,file in enumerate(os.listdir(dir)):

    filepath=os.path.join(dir,file)
    im=torch.from_numpy(tifffile.imread(filepath))
    if im.max()>maxfound:
        maxfound=im.max()

    '''
    ims[filenum%nims]=im

    if (filenum+1)%nims==0:
        fig,ax=plt.subplots(8,12)
        axf=ax.flatten()

        for i,ax in enumerate(axf):
            ax.imshow(ims[i],cmap="gray")
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    '''
print(maxfound)




