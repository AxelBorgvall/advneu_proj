import os

import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import tifffile
import myNets

def normalize_Hela(tim):
    HIGHEST_VALUE = 36863.0
    LOWEST_VALUE = 32995.0

    tim = (tim.to(torch.float) - LOWEST_VALUE) / (HIGHEST_VALUE - LOWEST_VALUE)
    return tim


def find_positions(probmap,npart=50,partwidth=40):
    prob_np=probmap.squeeze().detach().cpu().numpy()
    prob_np-=np.min(prob_np)


    pos=np.zeros((npart,2))

    h,w=prob_np.shape
    x = np.arange(w,dtype=float)
    y = np.arange(h,dtype=float)
    xgrid, ygrid = np.meshgrid(x, y)

    for i in range(npart):
        flat_idx = np.argmax(prob_np)
        y, x = divmod(flat_idx, prob_np.shape[1])
        pos[i] = [x, y]
        rsqr=(ygrid-pos[i,1])**2+(xgrid-pos[i,0])**2


        prob_np*=1-0.4*np.exp(-rsqr/partwidth**2)

    return pos

def sqr_around(im,prob,pos,max_delta=30):
    #pos is x,y
    arr=prob.squeeze().detach().cpu().numpy()
    x=pos.astype(int)
    window=arr[max(0,x[1]-max_delta):min(arr.shape[0],x[1]+max_delta),max(0,x[0]-max_delta):min(arr.shape[1],x[0]+max_delta)]

    yarr=np.sum(window**2,axis=1)
    xarr = np.sum(window ** 2, axis=0)

    x1=np.argmin(xarr[:xarr.size//2])
    x2 = np.argmin(xarr[xarr.size // 2:])+xarr.size//2

    y1 = np.argmin(yarr[:yarr.size // 2])
    y2 = np.argmin(yarr[yarr.size // 2:]) + yarr.size // 2

    x1im=x1+x[0]-max_delta
    x2im = x2 + x[0] - max_delta

    y1im=y1+x[1]-max_delta
    y2im = y2 + x[1] - max_delta
    return im[y1im:y2im,x1im:x2im],window[y1:y2,x1:x2]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=myNets.Unet(1,1,layers=[64,128,256])
loc=myNets.Localizer(model=model, n_transforms=8)
optim=torch.optim.Adam(loc.parameters(),lr=0.0002)

loc.load_state_dict(torch.load("cell_localizer.pth"))

dir="Fluo-N2DL-HeLa/01"
writedir="lodestar_cropped_cell_images"

images=[]
probs=[]


for filenum,f in enumerate(os.listdir(dir)):
    filepath=os.path.join(dir,f)
    im=tifffile.imread(filepath)

    tim = torch.from_numpy(im).to(device)

    tim=normalize_Hela(tim)

    with torch.no_grad():
        #Compute prob
        pred = loc(tim[None, None, ...])

        #Find pos
        pos = find_positions(pred)
        for i in range(len(pos)):
            #Find squares around pos with small border values
            cropim,cropprob=sqr_around(im,pred,pos[i])

            # See of find_pos with few particles give different pos
            checkpos = find_positions(torch.from_numpy(cropprob), npart=3)
            std=np.std(checkpos,axis=0)
            if np.all(std<5):
                images.append(cropim)
                probs.append(cropprob)
            else:
                print("thrown due to multiple")
                plt.imshow(cropim)
                plt.show()
            #remove accordingly

        #check that images are not overly mutually similar

        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                a, b = images[i], images[j]
                if a.shape == b.shape:
                    if np.allclose(a,b,atol=10):
                        print(f"Matrices {i} and {j} are similar.")

        #plot on big grid to see if it looks swanky




        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flat if n > 1 else [axes]

        for ax, img in zip(axes, images):
            ax.imshow(img, cmap='gray')
            ax.axis('off')

        for ax in axes[n:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        for i in range(len(images)):
            tifffile.imwrite(os.path.join(writedir,f"im{filenum}_cell{i}.tif"),images[i])
    images=[]
    probs=[]