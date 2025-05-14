import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import torch
from myClasses import myNets,myDataSets

# ——— Model setup (assume yours is correct) ——————————————————————————

device = torch.device("cpu")
imgloss = myNets.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.mse_loss)
enc = myNets.ConvDown(1, 64, [32, 64, 64], doubleconv=True, batchnorm=True)
dec = myNets.ConvUp(64, 1, [64, 64, 32], doubleconv=True, last_act_sig=False)
model = myNets.VQ_VAE(enc, dec, (1, 64, 64), imageloss=imgloss, num_embeddings=700)
model.load_state_dict(torch.load("../state_dicts/VQ_VAE_small_sd1.pth", map_location=device))
model.to(device).eval()

grid_width=8

K, D = model.vq.n_emb, model.vq.emb_dim
codebook = model.vq.embedding.weight.detach().cpu().clone()  # (K, D)

dataset=myDataSets.VaeDataset("../data/VAE_single_cell2_rotated","angles.npy")
dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)

def decode(z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.decoder(z)

def encode(z:torch.Tensor)->torch.Tensor:
    with torch.no_grad():
        return model.get_indices(z)

# ——— GUI setup —————————————————————————————————————————————

class VQVAEEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VQ-VAE Codebook Editor")

        # Controls: Load, Reset, Reconstruct
        ctrl_frame = ttk.Frame(self)
        ttk.Button(ctrl_frame, text="Load Image", command=self.load_image).pack(side="left", padx=5)
        ttk.Button(ctrl_frame, text="Reset Indices", command=self.reset_indices).pack(side="left", padx=5)
        ttk.Button(ctrl_frame, text="Reconstruct", command=self.reconstruct).pack(side="left", padx=5)
        ctrl_frame.pack(pady=5)

        # 8×8 grid of Int entries
        self.entries = []
        grid_frame = ttk.Frame(self)
        for i in range(grid_width):
            row = []
            for j in range(grid_width):
                var = tk.IntVar(value=-1)
                ttk.Entry(grid_frame, width=4, textvariable=var, justify='center').grid(row=i, column=j, padx=2, pady=2)
                row.append(var)
            self.entries.append(row)
        grid_frame.pack(padx=10, pady=10)

        # Canvases for original and reconstructed images
        canvas_frame = ttk.Frame(self)
        self.canvas_orig = tk.Canvas(canvas_frame, width=256, height=256)
        self.canvas_orig.pack(side='left', padx=5)
        self.canvas_recon = tk.Canvas(canvas_frame, width=256, height=256)
        self.canvas_recon.pack(side='left', padx=5)
        canvas_frame.pack(padx=10, pady=10)
        self.orig_on_canvas = None
        self.recon_on_canvas = None

    def reconstruct(self):
        # Build latent (1, D, 8, 8)
        latent = np.zeros((1, D, grid_width, grid_width), dtype=np.float32)
        for i in range(grid_width):
            for j in range(grid_width):
                idx = self.entries[i][j].get()
                if 0 <= idx < K:
                    latent[0, :, i, j] = codebook[idx].numpy()
        # Decode
        z = torch.from_numpy(latent).to(device)
        out = decode(z)                  # shape (1, 1, H, W)
        img = out[0, 0].cpu().numpy()    # (H, W)

        # Normalize to [0,255]
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        img = (img * 255).astype(np.uint8)
        pil = Image.fromarray(img).resize((256,256), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(pil)

        # Update reconstructed canvas
        if self.recon_on_canvas is None:
            self.recon_on_canvas = self.canvas_recon.create_image(0,0, anchor='nw', image=tk_img)
        else:
            self.canvas_recon.itemconfig(self.recon_on_canvas, image=tk_img)
        self.canvas_recon.image = tk_img

    def reset_indices(self):
        # Reset all entries to -1
        for row in self.entries:
            for var in row:
                var.set(-1)

    def load_image(self):
        x,ag=next(iter(dataloader))
        try:
            # Convert the tensor image `x` to a format suitable for display in Tkinter
            x_np = x.squeeze().cpu().numpy()  # Convert to numpy array and remove the singleton dimension
            x_resized = np.resize(x_np, (64, 64))  # Resize to 64x64 (though it's already 64x64, this is more general)

            # Normalize to [0, 255] (convert float values between 0 and 1 to uint8)
            x_resized = (x_resized * 255).astype(np.uint8)

            # Convert the numpy array to a PIL image
            pil_image = Image.fromarray(x_resized)

            # Convert the PIL image to a Tkinter-compatible format
            tk_image = ImageTk.PhotoImage(pil_image)

            # Display the image on the canvas
            if self.orig_on_canvas is None:
                self.orig_on_canvas = self.canvas_orig.create_image(0, 0, anchor='nw', image=tk_image)
            else:
                self.canvas_orig.itemconfig(self.orig_on_canvas, image=tk_image)

            # Keep a reference to the image object to prevent it from being garbage collected
            self.canvas_orig.image = tk_image

            indices=model.get_indices(x)
            for i in range(grid_width):
                for j in range(grid_width):
                    index_value = indices[
                        0, 0, i, j].item()  # Get the index value from the tensor (convert to Python scalar)
                    if 0 <= index_value < K:  # Make sure the index is valid
                        self.entries[i][j].set(index_value)  # Set the grid entry to the corresponding index

        except Exception as e:
            print(f"Error loading image: {e}")
            # Placeholder if `x` is not set or any error occurs
            pass

if __name__ == "__main__":
    app = VQVAEEditor()
    app.mainloop()