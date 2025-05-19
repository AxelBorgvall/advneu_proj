import os

import numpy as np
import matplotlib.pyplot as plt


x=np.arange(700)


rootdir="../data/classified_cells"

#Want to load each count, normalize with number of ims in dict
ar=[]
names=[]
for i,d in enumerate(os.listdir(rootdir)):
    filename = "VQ_VAE_small_sd2_" + d + "_occurences.npy"
    ar.append(np.load("../data/codebook_occurences/"+filename).astype(float))
    norm=0
    for f in os.listdir(os.path.join(rootdir,d)):
        norm+=1
    ar[i]/=norm
    names.append(d)


for i in range(len(ar)):
    plt.bar(x,ar[i],label=names[i])

plt.legend()
plt.yscale('log')
plt.show()


