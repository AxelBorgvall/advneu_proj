import torch
import myNets
import myDataSets
from torch.utils.data import DataLoader



import matplotlib.pyplot as plt

# Instantiate your dataset
dataset = myDataSets.SingleCellDataset("SingleParticleImages/interphase-control",120,repeat=5)

# Wrap it in a DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model=myNets.Unet(1,1,layers=[64,128,256])
loc=myNets.Localizer(model=model, n_transforms=8)
optim=torch.optim.Adam(loc.parameters(),lr=0.0002)
myNets.train_localizer(loc=loc, dataloader=loader, optimizer=optim, epochs=30, filename="cell_localizer.pth")

torch.save(loc.state_dict(),"cell_localizer.pth")
