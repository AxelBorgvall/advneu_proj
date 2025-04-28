import torch
import torch.nn as nn
import torch.nn.functional as F
import myNets



'''
ToDo:
1. Rewrite Unet to be a little nicer. Take a general list of channel numbers as input yea. Check!
2. Write a lodeStar wrapper class using a Unet network. Needs to initialize and overwrite training step
-Need to rewrite cost. theres no way it's correct mane. 

3. Create dataset pipeline. Make it general with options for classes?
-start basic test single particle detection
-try generalizing to multiple
-try retraining with blank dist for edge cases, like particles overlapping. 

5. If it works try to make networks to distinguish cell types
'''
