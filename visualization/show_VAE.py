import matplotlib.pyplot as plt
import torch
from myClasses import myDataSets, myNets
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imgloss= myNets.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.l1_loss)
model= myNets.VAE(inputshape=(64, 64), latent_dim=344, convchannels=[64, 128, 256], fc_layers=[2048, 1024], beta=1.0)
dataset= myDataSets.VaeDataset("../data/VAE_single_cell2_noise", "angles.npy")

loader=DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    #num_workers=1,
    pin_memory=True
)

model.load_state_dict(torch.load("../state_dicts/comp_VAE_sd_1.pth"))

model.eval()

images,angles=next(iter(loader))

for i in range(len(images)):
    with torch.no_grad():
        im=images[i].unsqueeze(0)
        ag = angles[i].unsqueeze(0)
        pred,mu,logvar=model(im)

        fig,axs=plt.subplots(1,2)
        axs[0].imshow(pred.cpu().squeeze(),cmap="gray")
        axs[1].imshow(im.cpu().squeeze(), cmap="gray")
        plt.show()

images,angles=next(iter(loader))

for i in range(len(images)):
    with torch.no_grad():
        im=images[i].unsqueeze(0)
        ag = angles[i].unsqueeze(0)
        pred,mu,logvar=model(im)

        fig,axs=plt.subplots(1,2)
        axs[0].imshow(pred.cpu().squeeze(),cmap="gray")
        axs[1].imshow(im.cpu().squeeze(), cmap="gray")
        plt.show()

