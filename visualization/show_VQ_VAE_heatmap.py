import numpy as np
import matplotlib.pyplot as plt


ar=np.load("../data/VQ_VAE_small_sd1_occurences.npy")

x=np.arange(len(ar))


plt.bar(x,ar)
plt.yscale('log')
plt.show()


