import torch
from myClasses import myDataSets, myNets
from torch.utils.data import DataLoader

imgloss= myNets.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.l1_loss)
model= myNets.VAE(inputshape=(64, 64), latent_dim=344, convchannels=[64, 128, 256], fc_layers=[2048, 1024], beta=1.0)
#model=myNets.VAE(inputshape=(64,64),latent_dim=30,convchannels=[16,32,64],fc_layers=[1024,512,256],beta=1.0,imageloss=imgloss)

dataset= myDataSets.VaeDataset("../data/VAE_single_cell2_rotated", "angles.npy")

loader=DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
if __name__=="__main__":
    myNets.train_vae(model, loader, optimizer, epochs=10)
    torch.save(model.state_dict(), "../state_dicts/comp_VAE_sd_1.pth")







