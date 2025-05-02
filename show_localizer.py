import os

import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import myNets

def normalize_Hela(tim):
    HIGHEST_VALUE = 36863.0
    LOWEST_VALUE = 32995.0

    tim = (tim.to(torch.float) - LOWEST_VALUE) / (HIGHEST_VALUE - LOWEST_VALUE)
    return tim

def laplacian(tim):
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32,device=tim.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

    # Apply convolution (padding=1 to keep same size)
    laplacian = torch.nn.functional.conv2d(tim, laplacian_kernel, padding=1)

    return laplacian

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=myNets.Unet(1,1,layers=[64,128,256])
loc=myNets.Localizer(model=model, n_transforms=8)
optim=torch.optim.Adam(loc.parameters(),lr=0.0002)

loc.load_state_dict(torch.load("cell_localizer.pth"))

dir="Fluo-N2DL-HeLa/01"

for f in os.listdir(dir):
    filepath=os.path.join(dir,f)
    im=tifffile.imread(filepath)

    tim = torch.from_numpy(im).to(device)

    tim=normalize_Hela(tim)

    with torch.no_grad():
        pred = loc(tim[None, None, ...])

        lap=laplacian(pred)

        print(pred.min())
        print(pred.max())
        print(torch.mean(lap))
        print(torch.std(lap))

        pos=find_positions(pred)

        fig, axs = plt.subplots(1, 2)
        axs[0].set_title("Input image")
        axs[0].imshow(tim.to("cpu").squeeze(), cmap="gray")

        axs[1].set_title("Model Prediciton")
        axs[1].imshow(pred.to("cpu").squeeze(), cmap="gray")
        axs[1].scatter(pos[:,0],pos[:,1],s=1,c='r')

        #axs[2].set_title("Prediction laplacian")
        #axs[2].imshow(lap.to("cpu").squeeze(), cmap="gray")

        plt.show()



