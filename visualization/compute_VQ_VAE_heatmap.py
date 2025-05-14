import matplotlib.pyplot as plt
import torch
from myClasses import myDataSets, myNets
from torch.utils.data import DataLoader
import numpy as np
def empirical_entropy(samples):
    samples=np.array(samples.detach().cpu())
    values, counts = np.unique(samples, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgloss = myNets.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.mse_loss)
    enc = myNets.ConvDown(1, 64, [32, 64, 64], doubleconv=True, batchnorm=True)
    dec = myNets.ConvUp(64, 1, [64, 64, 32], doubleconv=True, last_act_sig=False)
    model = myNets.VQ_VAE(enc, dec, (1, 64, 64), imageloss=imgloss, num_embeddings=700)
    # model=myNets.VAE(inputshape=(64,64),latent_dim=30,convchannels=[16,32,64],fc_layers=[1024,512,256],beta=1.0,imageloss=imgloss)

    dataset = myDataSets.VaeDataset("../data/VAE_single_cell2_rotated", "angles.npy")
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model.load_state_dict(torch.load("../state_dicts/VQ_VAE_small_sd1.pth"))

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model=model.to(device)

    index_range=torch.arange(0,model.vq.n_emb,device=device)
    count=torch.zeros_like(index_range,device=device)

    for images, angles in loader:
        with torch.no_grad():
            images = images.to(device)

            indices = model.get_indices(images).flatten()
            counts_batch = torch.bincount(indices, minlength=model.vq.n_emb)
            count += counts_batch

    count=count.detach().cpu()

    car=np.array(count)
    np.save("../data/VQ_VAE_small_sd1_occurences.npy",car)

