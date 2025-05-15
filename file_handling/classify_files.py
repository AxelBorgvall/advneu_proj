import os
import sys
import tifffile
import torch
import matplotlib.pyplot as plt


nims=8*12
dir= "../data/VAE_single_cell2_rotated"

ims=torch.zeros((nims,64,64))


normal=[0,1,2,3,4,5,9,10,11,12,14,15,17,386,392,415,453,460,587,710]
elongated=[6,18,36,38,71,105,134,152,343,354,1098,1074,1193]
metaphase=[63,76,85,93,98,146,157,166,172,399,674,685,828,819,1347]
prometaphase=[39,128,206,303,517,1192,1291,1276,1672,1825,2536,3039,3451,4060,4707,5143]
grape=[632,786,1332]
anaphase=[994,2472,4535,5690,6163,6559,6600,7563,9134,10491,12213]
map=[1571,2310,2573,7640,8257,15119,15301,15660,15836,16099]
binuclear=[2025,2198,2352,2362,2479,2465,2417,2559,2571,2657,2630,13673,14174,14208]
death=[3255,3242,3360,3421,3549,3681,3853,4126,4272,4346,4343,6328,12892,16711]
condensed=[4662,4656,6525,8939,9290,11125,11081,11564,11751,11933,12205,12567,12655,13095]

names=["normal","elongated","metaphase","prometaphase","grape","anaphase","map","binuclear","death","condensed"]

classes=[normal,elongated,metaphase,prometaphase,grape,anaphase,map,binuclear,death,condensed]

for filenum,file in enumerate(os.listdir(dir)):
    filepath=os.path.join(dir,file)
    im=tifffile.imread(filepath)
    #im=torch.from_numpy(tifffile.imread(filepath))

    for i in range(len(names)):
        if filenum in classes[i]:
            dest_dir = os.path.join("../data/classified_cells", names[i])
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, file)
            tifffile.imwrite(dest_path, im)
            break  # Stop after finding the first matching class


    '''
    ims[filenum%nims]=im

    if (filenum+1)%nims==0:
        fig,ax=plt.subplots(8,12)
        axf=ax.flatten()

        for i,ax in enumerate(axf):
            ax.imshow(ims[i],cmap="gray")
            ax.axis('off')
            ax.set_title(f"{filenum - nims + 1 + i}", fontsize=6, pad=2)
        plt.tight_layout()
        plt.show()
    '''
