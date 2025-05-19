import torch
from myClasses import myDataSets, myNets
from torch.utils.data import DataLoader
if __name__=="__main__":
    imgloss = myNets.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.mse_loss)
    enc = myNets.ConvDown(1, 64, [32, 64, 64], doubleconv=True, batchnorm=True)
    dec = myNets.ConvUp(64, 1, [64, 64, 32], doubleconv=True, last_act_sig=False)
    model = myNets.VQ_VAE(enc, dec, (1, 64, 64), imageloss=imgloss, num_embeddings=700, codebook_refresh_period=-1,codebook_usage_threshold=1)
    #model=myNets.VAE(inputshape=(64,64),latent_dim=30,convchannels=[16,32,64],fc_layers=[1024,512,256],beta=1.0,imageloss=imgloss)

    dataset= myDataSets.VaeDataset("../data/VAE_single_cell2_rotated", "angles.npy")

    loader=DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    model.load_state_dict(torch.load("../state_dicts/VQ_VAE_small_sd2.pth"))
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

    myNets.train_vq_vae(model, loader, optimizer, epochs=10)
    torch.save(model.state_dict(), "../state_dicts/VQ_VAE_small_sd2.pth")

