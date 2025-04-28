import sys
import tifffile
import torch
import matplotlib.pyplot as plt



myims=[]

i=0
offset=0

dir="SingleParticleImages"

def handle_a(list,tensor):
    try:
        print("Give coords to crop on form: x1,x2,y1,y2")
        s = input()
        nums = [int(x.strip()) for x in s.split(",")]
        imidsh = tensor[nums[0]:nums[1], nums[2]:nums[3]]
        list.append(imidsh)
    except Exception as e:
        print(e)
    return


def handle_q(list,tensor):
    sys.exit()

def handle_qs(list,tensor):
    for i in range(len(list)):
        torch.save(list[i],f"SingleParticleImages/singlecell_nr_{i+offset}")
    sys.exit()
def handle_c(list,tensor):
    return

def handle_default(list,tensor):
    print("Default")

switch = {
    "q": handle_q,
    "qs": handle_qs,
    "a":handle_a,
    "c":handle_c
}
while True:
    filename = f'Fluo-N2DL-HeLa/01/t{i+offset:03}.tif'
    array = tifffile.imread(filename)
    tensor = torch.from_numpy(array)
    if tensor.dtype == torch.uint16:
        tensor = tensor.float() / 65535.0
        min = tensor.min()
        max = tensor.max()
        tensor = (tensor - min) / (max - min)
    fig, ax = plt.subplots()
    ax.imshow(tensor, cmap='gray')
    ax.axis('off')

    plt.show(block=False)  # Non-blocking display
    plt.pause(0.2)
    while True:
        print(f"Viewing image {i}, \"a\" to crop and append,  \"q\" to quit without saving, \"qs\" to quit and save,  \"c\" to continue to next image")
        s=input()
        switch.get(s, handle_default)(myims,tensor)  # calls correct function
        if s == "c":
            plt.close(fig)  # Close the figure before continuing to next image
            break
    i+=1