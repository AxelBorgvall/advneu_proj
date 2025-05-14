import torch
from myClasses import myDataSets, myNets
from torch.utils.data import DataLoader

if __name__=="__main__":
    imgloss = myNets.DiffImageLoss(scaling=2, norm=True, lossfunc=torch.nn.functional.l1_loss)
    model = myNets.ME_VAE(inputshape=(64, 64), latent_dim=30, convchannels=[32, 64, 128], fc_layers=[1024, 512],
                          num_encoders=4, beta=1, imageloss=imgloss)
    # model=myNets.VAE(inputshape=(64,64),latent_dim=30,convchannels=[16,32,64],fc_layers=[1024,512,256],beta=1.0,imageloss=imgloss)

    dataset = myDataSets.VaeDataset("../data/VAE_single_cell2_noise", "angles.npy")

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    myNets.train_vae(model, loader, optimizer, epochs=30)
    torch.save(model.state_dict(), "cell_ME_VAE_big_b1_s2_t1.pth")



