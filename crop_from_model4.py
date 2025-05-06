import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
import myNets
from skimage.feature import blob_log
from scipy.ndimage import maximum_filter, label
import torch.nn.functional as F
from scipy.optimize import curve_fit
from kornia import morphology as morph


def slivermask(img: torch.Tensor, sliver_width: int):

    # build a square kernel of side = 2*sliver_width+1
    kernel = torch.ones((2*sliver_width+1, 2*sliver_width+1),
                        dtype=img.dtype, device=img.device)
    # first erode, then dilate
    opened = morph.opening(img, kernel)

    return opened>=torch.mean(opened)

def remove_slivers(img: torch.Tensor,width=3):
    mask=slivermask(img,width)
    im=torch.zeros_like(img)
    im[mask]=img[mask]
    return im
def expand_true_neighbors(mask):
    """
    Expands True regions in a boolean matrix to adjacent False values.

    Args:
    - mask (torch.Tensor): A boolean 2D tensor.

    Returns:
    - torch.Tensor: Updated mask with False values touching True set to True.
    """
    mask = mask.float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)  # 3x3 kernel

    # Perform 2D convolution (binary dilation)
    dilated = F.conv2d(mask, kernel, padding=1)
    expanded = (dilated > 0).squeeze(0).squeeze(0)  # Any neighboring True makes it True

    return expanded.bool()



def normder(tensor):
    device = tensor.device  # Get device of input

    xker = torch.tensor([[-1,1],
                        [-1,1]], dtype=torch.float32, device=device).view(1,1,2,2)
    yker = torch.tensor([[1, 1],
                         [-1, -1]], dtype=torch.float32, device=device).view(1,1,2,2)

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)  # (B, 1, H, W)

    der = (F.conv2d(tensor, xker, padding=0)**2)+(F.conv2d(tensor, yker, padding=0)**2)
    der = F.pad(der, (0, 1, 0, 1), mode='constant', value=0)
    return der
def laplacian_2d(tensor):
    device = tensor.device  # Get device of input

    kernel = torch.tensor([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=torch.float32, device=device)
    kernel = kernel.view(1, 1, 3, 3)  # shape (out_channels, in_channels, H, W)

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)  # (B, 1, H, W)

    lap = F.conv2d(tensor, kernel, padding=1)
    lap=lap.squeeze()
    lap[:,0]=0
    lap[:,-1]=0
    lap[-1,:]=0
    lap[0,:]=0

    return lap
def normalize_Hela(tim):
    HIGHEST_VALUE = 36863.0
    LOWEST_VALUE = 32995.0

    tim = (tim.to(torch.float) - LOWEST_VALUE) / (HIGHEST_VALUE - LOWEST_VALUE)
    return tim

def crop_blob(image, center_x, center_y, radius, crop_size=64):
    h, w = image.shape
    half_crop = crop_size // 2

    # Ensure center is int
    cx, cy = int(round(center_x)), int(round(center_y))

    # Compute crop box
    x1 = max(cx - half_crop, 0)
    x2 = min(cx + half_crop, w)
    y1 = max(cy - half_crop, 0)
    y2 = min(cy + half_crop, h)

    crop = torch.zeros((crop_size, crop_size), dtype=image.dtype)
    valid_crop = image[y1:y2, x1:x2]

    # Paste into center of empty canvas
    y_offset = half_crop - (cy - y1)
    x_offset = half_crop - (cx - x1)
    crop[y_offset:y_offset+valid_crop.shape[0], x_offset:x_offset+valid_crop.shape[1]] = valid_crop
    return crop

def trim(im,prob):
    #find contour

    x=torch.arange(64)-32
    y = torch.arange(64) - 32
    x,y=torch.meshgrid(x,y)

    r=torch.sqrt(x**2+y**2)

    der=normder(prob).squeeze()


    mod_der=der/(1.0*r**1.7+10)
    mod_der+=torch.exp(-r**2/270)*torch.mean(mod_der)*3


    thresh=torch.mean(mod_der)*1.2
    imtrim=torch.zeros_like(im)
    imtrim[mod_der>thresh]=im[mod_der>thresh]

    return imtrim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=myNets.Unet(1,1,layers=[64,128,256])
loc=myNets.Localizer(model=model, n_transforms=8)
optim=torch.optim.Adam(loc.parameters(),lr=0.0002)

loc.load_state_dict(torch.load("cell_localizer.pth"))

dir="Fluo-N2DL-HeLa/02"
writedir="VAE_single_cell2"

images=[]
probs=[]

for filenum,f in enumerate(os.listdir(dir)):
    if filenum<=-1:
        continue

    filepath=os.path.join(dir,f)
    im=tifffile.imread(filepath)

    tim = torch.from_numpy(im).to(device)

    tim=normalize_Hela(tim)
    with torch.no_grad():
        #Compute prob
        pred = loc(tim[None, None, ...])


        plotpred=pred.squeeze().to("cpu")
        plotim=tim.squeeze().to("cpu")

        blobs = blob_log(plotpred, max_sigma=10, threshold=0.3)

        rejects = []
        imblobs = []
        for y, x, r in blobs:
            probcrop = crop_blob(plotpred, x, y, r)
            imcrop=crop_blob(plotim, x, y, r)
            imtrimmed=trim(imcrop,probcrop)

            mask = imtrimmed == 0.0
            mask = expand_true_neighbors(mask)

            imtrimmed=remove_slivers(imtrimmed[None,None,...],3).squeeze()





            if not normder(imtrimmed).squeeze()[mask].max()>torch.mean(imtrimmed)**1.8*36:
                imblobs.append(imtrimmed)
            else:
                rejects.append(imtrimmed)


        '''
        print(len(imblobs))
        n1 = max(int(np.sqrt(len(imblobs))), 1)
        n2 = int(len(imblobs) // n1 + 1)
        fig, axs = plt.subplots(n1, n2)
        axflat = axs.flatten()
        for i, ax in enumerate(axflat):
            if i < len(imblobs):
                ax.imshow(imblobs[i], cmap="gray")
            ax.axis("off")  # hide axes regardless
        plt.tight_layout()
        plt.show()

        print(len(rejects))
        n1 = max(int(np.sqrt(len(rejects))), 1)
        n2 = int(len(rejects) // n1 + 1)
        fig, axs = plt.subplots(n1, n2)
        axflat = axs.flatten()
        for i, ax in enumerate(axflat):
            if i < len(rejects):
                ax.imshow(rejects[i], cmap="gray")
            ax.axis("off")  # hide axes regardless
        plt.tight_layout()
        plt.show()
        '''
        print(f"\rfilenum: {filenum}",end="")
        for i in range(len(imblobs)):
            filename=f"02_image{filenum}_cell{i}.tif"
            filepath=os.path.join(writedir,filename)
            np_img = imblobs[i].detach().cpu().numpy()
            tifffile.imwrite(filepath, np_img.astype(np.float32))


