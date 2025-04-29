import numpy as np
import torch
import numpy
import tifffile
import os

dir="SingleParticleImages/interphase-control"

for filename in os.listdir(dir):
    filepath=os.path.join(dir,filename)
    if os.path.isfile(filepath):
        tens=torch.load(filepath)
        activation=((tens.max()-tens.min())*0.2+tens.min()).item()
        arr=np.array(tens,dtype=float)

        xarr=np.all(arr<activation,axis=1)
        yarr=np.all(arr<activation,axis=0)

        xdiffs=np.diff


