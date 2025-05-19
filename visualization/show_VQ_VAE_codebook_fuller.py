import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import torch
from myClasses import myNets, myDataSets

# ——— Model setup (assume yours is correct) ——————————————————————————

device = torch.device("cpu")
imgloss = myNets.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.mse_loss)
enc = myNets.ConvDown(1, 64, [32, 64, 64], doubleconv=True, batchnorm=True)
dec = myNets.ConvUp(64, 1, [64, 64, 32], doubleconv=True, last_act_sig=False)
model = myNets.VQ_VAE(enc, dec, (1, 64, 64), imageloss=imgloss, num_embeddings=700)
model.load_state_dict(torch.load("../state_dicts/VQ_VAE_small_sd2.pth", map_location=device))
model.to(device).eval()

grid_width = 8
K, D = model.vq.n_emb, model.vq.emb_dim
codebook = model.vq.embedding.weight.detach().cpu().clone()  # (K, D)

dataset = myDataSets.VaeDataset("../data/VAE_single_cell2_rotated", "angles.npy")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


def decode(z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.decoder(z)


def encode(z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.get_indices(z)


class VQVAEEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VQ-VAE Codebook Editor")

        # Controls: Random Load, Specific Load, Reset, Reconstruct
        ctrl_frame = ttk.Frame(self)
        # Random image loader
        ttk.Button(ctrl_frame, text="Load Random Image", command=self.load_random_image).pack(side="left", padx=5)
        # Specific index loader
        self.specific_index = tk.IntVar(value=0)
        ttk.Label(ctrl_frame, text="Index:").pack(side="left")
        ttk.Entry(ctrl_frame, width=4, textvariable=self.specific_index, justify='center').pack(side="left", padx=2)
        ttk.Button(ctrl_frame, text="Load Image", command=self.load_specific_image).pack(side="left", padx=5)
        # Reset and reconstruct
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

    def display_image_and_indices(self, x_tensor):
        # Display image
        x_np = x_tensor.squeeze().cpu().numpy()
        x_np = (x_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(x_np).resize((256, 256), Image.NEAREST)
        tk_image = ImageTk.PhotoImage(pil_image)
        if self.orig_on_canvas is None:
            self.orig_on_canvas = self.canvas_orig.create_image(0, 0, anchor='nw', image=tk_image)
        else:
            self.canvas_orig.itemconfig(self.orig_on_canvas, image=tk_image)
        self.canvas_orig.image = tk_image

        # Load and set code indices
        indices = model.get_indices(x_tensor)
        for i in range(grid_width):
            for j in range(grid_width):
                val = indices[0, 0, i, j].item()
                self.entries[i][j].set(val)

    def load_random_image(self):
        x, _ = next(iter(dataloader))
        self.display_image_and_indices(x)

    def load_specific_image(self):
        idx = self.specific_index.get()
        try:
            x, _ = dataset[idx]
            # Add batch dimension
            x = x.unsqueeze(0)
            self.display_image_and_indices(x)
        except Exception as e:
            print(f"Error loading index {idx}: {e}")

    def reconstruct(self):
        latent = np.zeros((1, D, grid_width, grid_width), dtype=np.float32)
        for i in range(grid_width):
            for j in range(grid_width):
                idx = self.entries[i][j].get()
                if 0 <= idx < K:
                    latent[0, :, i, j] = codebook[idx].numpy()
        z = torch.from_numpy(latent).to(device)
        out = decode(z)
        img = out[0, 0].cpu().numpy()
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        img = (img * 255).astype(np.uint8)
        pil = Image.fromarray(img).resize((256, 256), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(pil)
        if self.recon_on_canvas is None:
            self.recon_on_canvas = self.canvas_recon.create_image(0, 0, anchor='nw', image=tk_img)
        else:
            self.canvas_recon.itemconfig(self.recon_on_canvas, image=tk_img)
        self.canvas_recon.image = tk_img

    def reset_indices(self):
        for row in self.entries:
            for var in row:
                var.set(-1)


if __name__ == "__main__":
    app = VQVAEEditor()
    app.mainloop()
