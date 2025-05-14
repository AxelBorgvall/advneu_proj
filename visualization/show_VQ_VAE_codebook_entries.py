import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import torch
from myClasses import myNets

# ——— Model setup (assume yours is correct) ——————————————————————————

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgloss = myNets.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.mse_loss)
enc = myNets.ConvDown(1, 64, [32, 64, 64], doubleconv=True, batchnorm=True)
dec = myNets.ConvUp(64, 1, [64, 64, 32], doubleconv=True, last_act_sig=False)
model = myNets.VQ_VAE(enc, dec, (1, 64, 64), imageloss=imgloss, num_embeddings=700)
model.load_state_dict(torch.load("../state_dicts/VQ_VAE_small_sd1.pth", map_location=device))
model.to(device).eval()

K, D = model.vq.n_emb, model.vq.emb_dim
codebook = model.vq.embedding.weight.detach().cpu().clone()  # (K, D)

def decode(z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.decoder(z)

# ——— GUI setup —————————————————————————————————————————————

class VQVAEEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VQ-VAE Codebook Editor")
        # 8×8 grid of Int entries
        self.entries = []
        grid_frame = ttk.Frame(self)
        for i in range(8):
            row = []
            for j in range(8):
                var = tk.IntVar(value=-1)
                e = ttk.Entry(grid_frame, width=4, textvariable=var, justify='center')
                e.grid(row=i, column=j, padx=2, pady=2)
                row.append(var)
            self.entries.append(row)
        grid_frame.pack(padx=10, pady=10)

        # Reconstruct button
        btn = ttk.Button(self, text="Reconstruct", command=self.reconstruct)
        btn.pack(pady=5)

        # Canvas for image
        self.canvas = tk.Canvas(self, width=256, height=256)
        self.canvas.pack(padx=10, pady=10)

        # Placeholder
        self.img_on_canvas = None

    def reconstruct(self):
        # Build latent (1, D, 8, 8)
        latent = np.zeros((1, D, 8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
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

        # Update canvas
        if self.img_on_canvas is None:
            self.img_on_canvas = self.canvas.create_image(0,0, anchor='nw', image=tk_img)
        else:
            self.canvas.itemconfig(self.img_on_canvas, image=tk_img)
        # keep a reference
        self.canvas.image = tk_img

if __name__ == "__main__":
    app = VQVAEEditor()
    app.mainloop()