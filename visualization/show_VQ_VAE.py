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

    #imgloss = myNets.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.mse_loss)
    #enc = myNets.ConvDown(1, 64, [32, 64], doubleconv=True, batchnorm=True)
    #dec = myNets.ConvUp(64, 1, [64, 32], doubleconv=True, last_act_sig=False)
    #model = myNets.VQ_VAE(enc, dec, (1, 64, 64), imageloss=imgloss, num_embeddings=2056)

    imgloss = myNets.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.mse_loss)
    enc = myNets.ConvDown(1, 64, [32, 64, 64], doubleconv=True, batchnorm=True)
    dec = myNets.ConvUp(64, 1, [64, 64, 32], doubleconv=True, last_act_sig=False)
    model = myNets.VQ_VAE(enc, dec, (1, 64, 64), imageloss=imgloss, num_embeddings=700)

    dataset = myDataSets.VaeDataset("../data/VAE_single_cell2_rotated", "angles.npy")
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model.load_state_dict(torch.load("../state_dicts/VQ_VAE_small_sd2.pth"))

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model=model.to(device)
    '''
    with torch.no_grad():
        samps=torch.zeros((50,16,1,8,8),device=device)
        for i in range(50):
            print(f"\r{i/50*100:.2f}%",end="")
            images, angles = next(iter(loader))

            images=images.to(device)
            angles=angles.to(device)

            ind=model.get_indices(images)
            samps[i]=ind.detach()
        print(empirical_entropy(samps.flatten()))
    '''
    model=model.to(torch.device("cpu"))
    images, angles = next(iter(loader))
    for i in range(len(images)):
        with torch.no_grad():
            im = images[i].unsqueeze(0)
            ag = angles[i].unsqueeze(0)
            pred, vq_loss = model(im)

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(pred.cpu().squeeze(), cmap="gray")
            axs[1].imshow(im.cpu().squeeze(), cmap="gray")
            plt.show()

    images, angles = next(iter(loader))

    for i in range(len(images)):
        with torch.no_grad():
            im = images[i].unsqueeze(0)
            ag = angles[i].unsqueeze(0)
            pred, vq_loss = model(im)

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(pred.cpu().squeeze(), cmap="gray")
            axs[1].imshow(im.cpu().squeeze(), cmap="gray")
            plt.show()




