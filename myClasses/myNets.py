import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import translate,rotate,warp_affine
from tqdm import tqdm

class DoubleConv(nn.Module):
    """(Conv => ReLU => Conv => ReLU)"""
    def __init__(self, in_channels, out_channels,act=nn.ReLU,norm=False):
        super().__init__()
        if not norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                act(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                act()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                act(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                act()
            )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels,scaling=2,act=nn.ReLU,norm=False):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(scaling),
            DoubleConv(in_channels, out_channels,act,norm)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True,scaling=2,act=nn.ReLU,norm=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scaling, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels,act,norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=scaling)
            self.conv = DoubleConv(in_channels, out_channels,act,norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad x1 if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DoubleConvUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,scaling=2):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128,scaling=scaling)
        self.down2 = Down(128, 256,scaling=scaling)
        self.down3 = Down(256, 512,scaling=scaling)
        self.down4 = Down(512, 1024,scaling=scaling)
        self.up1 = Up(1024 + 512, 512, bilinear,scaling=scaling)
        self.up2 = Up(512 + 256, 256, bilinear,scaling=scaling)
        self.up3 = Up(256 + 128, 128, bilinear,scaling=scaling)
        self.up4 = Up(128 + 64, 64, bilinear,scaling=scaling)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, layers=[64, 128, 256, 512], bilinear=True, scaling=2,act=nn.ReLU,norm=False):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.downs=nn.ModuleList()
        self.ups=nn.ModuleList()

        self.inc=DoubleConv(n_channels, layers[0],norm=norm)
        self.outc=nn.Sequential(nn.Conv2d(layers[0], n_classes, kernel_size=1),nn.ReLU())

        self.nlayers=len(layers)

        self.xlist=[None]*self.nlayers
        for i in range(self.nlayers-1):
            self.downs.append(Down(layers[i], layers[i + 1], scaling=scaling,act=act,norm=norm))
            self.ups.append(Up(layers[self.nlayers-1-i]+layers[self.nlayers-2-i],layers[self.nlayers-2-i],scaling=scaling,bilinear=bilinear,act=act,norm=norm))
        self.to(self.device)


    def forward(self, x):
        self.xlist[0]=self.inc(x)
        for i in range(self.nlayers-1):
            self.xlist[i+1]=self.downs[i](self.xlist[i])
        x=self.xlist[-1]
        for i in range(self.nlayers-1):
            x=self.ups[i](x,self.xlist[-i-2])
        return self.outc(x)


#LodeSTAR definition---------------------------------------------------------------------------------------------------------------------

def mass_centroid(tensor):
    # tensor: (B, C, H, W)
    B, C, H, W = tensor.shape

    device = tensor.device

    y_coords = torch.linspace(-H/2,H/2,steps=H, device=device).float()
    x_coords = torch.linspace(-W/2,W/2,steps=W, device=device).float()
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)


    x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    mass = tensor.sum(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
    mass = mass + 1e-8

    x_centroid = (tensor * x_grid).sum(dim=(-2, -1), keepdim=False) / mass.squeeze(-1).squeeze(-1)
    y_centroid = (tensor * y_grid).sum(dim=(-2, -1), keepdim=False) / mass.squeeze(-1).squeeze(-1)

    centroids = torch.stack((x_centroid, y_centroid), dim=-1)
    return centroids

def image_translation(batch,translation):

    return translate(batch,translation)

def inverse_translation(preds,applied_translation):
    return preds-applied_translation

def image_rotation(batch,angles):
    return rotate(batch,angles)

def inverse_rotation(preds,angles):
    cosines = torch.cos(angles*(torch.pi/180))
    sines = torch.sin(angles*(torch.pi/180))

    R = torch.stack([
        torch.stack([cosines, -sines], dim=1),
        torch.stack([sines, cosines], dim=1)
    ], dim=1)

    return torch.bmm(R, preds.unsqueeze(2)).squeeze(2)  # (n,2)

def image_affine_transform(image: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
    """
    Applies an affine transformation to an image with (0,0) at the center.

    Args:
        image: Tensor of shape (B, C, H, W)
        affine_matrix: Tensor of shape (B, 2, 3) in centered coordinates

    Returns:
        Transformed image of shape (B, C, H, W)
    """
    B, C, H, W = image.shape

    # Convert center-origin affine to Kornia format (normalized coordinates)
    center = torch.tensor([W / 2, H / 2], device=image.device).view(1, 1, 2)

    # Build full 3x3 matrices for translation math
    T_center = torch.eye(3, device=image.device).unsqueeze(0).repeat(B, 1, 1)
    T_center[:, 0, 2] = -center[:, 0, 0]
    T_center[:, 1, 2] = -center[:, 0, 1]

    T_uncenter = torch.eye(3, device=image.device).unsqueeze(0).repeat(B, 1, 1)
    T_uncenter[:, 0, 2] = center[:, 0, 0]
    T_uncenter[:, 1, 2] = center[:, 0, 1]

    # Add bottom row to affine_matrix to make it 3x3
    A = torch.cat([affine_matrix, torch.tensor([[[0., 0., 1.]]], device=image.device).repeat(B, 1, 1)], dim=1)

    # Combine transformations: uncenter @ A @ center
    A_total = T_uncenter @ A @ T_center
    A_total = A_total[:, :2, :]  # warp_affine expects (B, 2, 3)

    # Apply transformation
    return warp_affine(image, A_total, dsize=(H, W), align_corners=False)


def forward_warp(
        pts: torch.Tensor,  # [N,2], in centered coords
        affine_matrix: torch.Tensor) -> torch.Tensor:
    N = pts.shape[0]
    device = pts.device

    # make 3×3 from 2×3
    A = torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)  # [N,3,3]
    A[:, :2, :] = affine_matrix

    # to homogeneous
    hom = torch.cat([pts, torch.ones(N, 1, device=device)], dim=1)  # [N,3]

    # matrix‐vector
    out = (A @ hom.unsqueeze(-1)).squeeze(-1)  # [N,3]

    return out[:, :2]


def inverse_warp(
        pts: torch.Tensor,
        affine_matrix: torch.Tensor) -> torch.Tensor:
    N = pts.shape[0]
    device = pts.device

    # build 3×3 blocks
    A = torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)
    A[:, :2, :] = affine_matrix  # [N,3,3]

    # invert
    A_inv = torch.inverse(A)  # [N,3,3]

    # to homogeneous
    hom = torch.cat([pts, torch.ones(N, 1, device=device)], dim=1)  # [N,3]

    # apply
    out = (A_inv @ hom.unsqueeze(-1)).squeeze(-1)  # [N,3]

    return out[:, :2]




class Localizer(nn.Module):
    def __init__(self,model,n_transforms=8,**kwargs):
        super(Localizer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model=nn.Sequential(model,torch.nn.Sigmoid()).to(self.device)
        self.model = model.to(self.device)
        #self.loss=LodestarLoss(beta)
        self.n_transforms=n_transforms
        return
    def forward(self,x):
        return self.model(x.to(self.device))

    def forward_tranform(self,batch,translation,angles):
        transformed=image_translation(batch,translation)
        return image_rotation(transformed,angles)

    def inverse_tranform(self,pred,translation,angles):
        invpred=inverse_rotation(pred,angles)
        return inverse_translation(invpred,translation)

    def get_loss(self,image):
        b,c,h,w=image.shape
        #expanding for transform
        images=image.unsqueeze(1).expand(-1,self.n_transforms,-1,-1,-1).contiguous()

        #flattening to feed into network
        flat=images.view(b*self.n_transforms,c,h,w)

        #getting random args
        tr = torch.rand(b, self.n_transforms, 2, device=images.device) * h//3 - h//6  # translations
        ag = torch.rand(b, self.n_transforms, device=images.device) * 360  # angles
        #flatten random args
        tr_flat = tr.view(b * self.n_transforms, 2)
        ag_flat = ag.view(b * self.n_transforms, )

        #transform images, run model
        transform_im=self.forward_tranform(flat,tr_flat,ag_flat)
        pred_flat=self.model(transform_im)



        #preds=pred_flat.view(b,self.n_transforms,1,h,w)

        centroids_flat=mass_centroid(pred_flat)


        #invert transforms (do some flattening shit ig)
        invpred=self.inverse_tranform(centroids_flat.squeeze(),tr_flat,ag_flat)
        invpred = invpred.view(b, self.n_transforms, 2)

        # invpred: [B, T, 2]

        diffs = invpred[:, 1:, :] - invpred[:, :-1, :]  # [B, T-1, 2]
        mse_per_sample = torch.mean((diffs ** 2).sum(dim=-1), dim=1)  # [B]
        return mse_per_sample.sum()

class LocalizerClassifier(nn.Module):
    def __init__(self,model:nn.Module,n_transforms:int=8,baseoffset:int=120,affine_warp=0.4):
        super(LocalizerClassifier, self).__init__()
        self.model = model
        self.n_transforms=n_transforms
        self.offset=baseoffset
        self.warp=affine_warp
        return
    def forward(self,x):
        return self.model(x)


    def forward_transform(self, batch, translation, angles,affine,ignore):
        # batch: (N, 1, H, W)
        # ignore: (N, 1, h, w)
        # translation: (N, 2)

        _, _, H, W = batch.shape
        N, _, h, w = ignore.shape

        # Apply translation
        transformed = image_translation(batch, translation)

        # Compute offset where to paste the ignore image
        # self.offset should be a scalar or tensor like (2,) for max translation
        noise = torch.rand(N, 2, device=ignore.device) * (self.offset // 1.5)
        ignore_offset = translation - self.offset + noise
        ignore_offset[:, 0] += (H - h) // 2  # y
        ignore_offset[:, 1] += (W - w) // 2  # x

        # Round and convert to int
        ignore_offset = ignore_offset.round().to(dtype=torch.long)

        # Vectorized paste of ignore into transformed
        patch_y = torch.arange(h, device=ignore.device).view(1, h, 1).expand(N, h, w)
        patch_x = torch.arange(w, device=ignore.device).view(1, 1, w).expand(N, h, w)
        offset_y = ignore_offset[:, 0].view(-1, 1, 1)
        offset_x = ignore_offset[:, 1].view(-1, 1, 1)
        target_y = patch_y + offset_y
        target_x = patch_x + offset_x

        # Mask for clipping if necessary (optional)
        in_bounds = (target_y >= 0) & (target_y < H) & (target_x >= 0) & (target_x < W)

        # Flatten indices for scatter
        batch_idx = torch.arange(N, device=ignore.device).view(N, 1, 1).expand(N, h, w)
        channel_idx = torch.zeros_like(batch_idx)

        # Masked paste (ignore pixels out of bounds)
        transformed[batch_idx[in_bounds], channel_idx[in_bounds], target_y[in_bounds], target_x[in_bounds]] = ignore.squeeze(1)[in_bounds]

        transformed=image_affine_transform(transformed,affine)

        '''
        example_affine=torch.rand(N,2,3,device=transformed.device)*0.4-0.2+torch.tensor([[1,0,0],[0,1,0]],dtype=torch.float32,device=transformed.device).unsqueeze(0).repeat(N,1,1)
        #first affine then rotation
        output=image_affine_transform(transformed,example_affine)
        output=image_rotation(output, angles).cpu().detach()
        #first affine then rotation
        coords=forward_warp(translation,example_affine)
        coords=inverse_rotation(coords,-angles)

        for i in range(len(output)):
            plt.imshow(output[i].squeeze(),cmap="gray")
            plt.scatter(coords[i,0].cpu().detach().squeeze()+W//2,coords[i,1].cpu().detach().squeeze()+H//2)
            plt.show()
        assert 1==0
        '''

        return image_rotation(transformed, angles)



    def inverse_tranform(self,pred,translation,angles,affine):
        invpred=inverse_rotation(pred,angles)
        invpred=inverse_warp(invpred,affine)
        return inverse_translation(invpred,translation)

    def get_loss(self,detect,ignore):

        #THIS ALL NEEDS TO BE REWRITTEN

        b,c,H,W=detect.shape
        _,_,h,w=ignore.shape
        #expanding for transform
        images=detect.unsqueeze(1).expand(-1,self.n_transforms,-1,-1,-1).contiguous()
        ignores=ignore.unsqueeze(1).expand(-1,self.n_transforms,-1,-1,-1).contiguous()
        #flattening to feed into network
        flat=images.view(b*self.n_transforms,c,H,W)
        ignores=ignores.view(b*self.n_transforms,c,h,w)


        #getting random args
        tr = torch.rand(b, self.n_transforms, 2, device=images.device) * h//2
        ag = torch.rand(b, self.n_transforms, device=images.device) * 360  # angles
        af=torch.rand(b,self.n_transforms,2,3,device=images.device)*self.warp-self.warp/2+\
                       torch.tensor([[1,0,0],[0,1,0]],dtype=torch.float32,device=images.device).unsqueeze(0).unsqueeze(0).repeat(b,self.n_transforms,1,1)

        #flatten random args
        tr_flat = tr.view(b * self.n_transforms, 2)
        ag_flat = ag.view(b * self.n_transforms, )
        af_flat=af.view(b*self.n_transforms,2,3)

        #transform images, run model
        transform_im=self.forward_transform(flat,tr_flat,ag_flat,af_flat,ignores)
        pred_flat=self.model(transform_im)


        centroids_flat=mass_centroid(pred_flat)



        #invert transforms (do some flattening shit ig)
        invpred=self.inverse_tranform(centroids_flat.squeeze(),tr_flat,ag_flat,af_flat)
        invpred = invpred.view(b, self.n_transforms, 2)

        # invpred: [B, T, 2]

        diffs = invpred[:, 1:, :] - invpred[:, :-1, :]  # [B, T-1, 2]
        mse_per_sample = torch.mean((diffs ** 2).sum(dim=-1), dim=1)  # [B]

        return mse_per_sample.sum()






def train_localizer(loc, dataloader, optimizer, epochs=300, filename="filename"):
    loc.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loc = loc.to(device)
    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for inputs in dataloader:
                inputs = inputs.to(device)  # [B, C, H, W]

                optimizer.zero_grad()
                loss = loc.get_loss(inputs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)  # sum up batch loss

            avg_loss = epoch_loss / len(dataloader.dataset)
            print(f"Epoch {epoch:3d}/{epochs}, avg loss: {avg_loss:.4f}")


    except KeyboardInterrupt:
        print("\n Training manually quit")
    finally:
        torch.save(loc.state_dict(), filename)
        print(f"Model saved to {filename}")

def train_localizer_classifier(loc, dataloader, optimizer, epochs=300, filename="filename"):
    loc.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loc = loc.to(device)

    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0

            # Wrap the dataloader in tqdm for batch-level progress
            pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}/{epochs}", leave=False)
            for images, ignores in pbar:
                images = images.to(device)
                ignores = ignores.to(device)

                optimizer.zero_grad()
                loss = loc.get_loss(images, ignores)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item() * images.size(0)
                epoch_loss += batch_loss

                avg_loss = epoch_loss / len(dataloader.dataset)
                pbar.set_postfix(loss=avg_loss)

            print(f"Epoch {epoch:3d}/{epochs}, avg loss: {avg_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining manually quit")
    finally:
        torch.save(loc.state_dict(), filename)
        print(f"Model saved to {filename}")
'''
def train_localizer_classifier(loc, dataloader, optimizer, epochs=300, filename="filename"):
    loc.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loc = loc.to(device)
    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for images,ignores in dataloader:
                images= images.to(device)
                ignores =ignores.to(device)  # [B, C, H, W]


                optimizer.zero_grad()
                loss = loc.get_loss(images,ignores)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)  # sum up batch loss

            avg_loss = epoch_loss / len(dataloader.dataset)
            print(f"Epoch {epoch:3d}/{epochs}, avg loss: {avg_loss:.4f}")


    except KeyboardInterrupt:
        print("\n Training manually quit")
    finally:
        torch.save(loc.state_dict(), filename)
        print(f"Model saved to {filename}")
'''



#VAE--------------------------------------------------------------------------------------------

class ReshapeLayer(nn.Module):
    def __init__(self, channels, height, width):
        super(ReshapeLayer, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        #reshape the tensor back to [batch_size, channels, height, width]
        return x.view(-1, self.channels, self.height, self.width)

def normalize_tensor(tens):
    return (tens-torch.mean(tens,dim=(-1,-2)).view(-1,1,1,1))/torch.std(tens,dim=(-1,-2)).view(-1,1,1,1)

class DiffImageLoss(nn.Module):
    def __init__(self,scaling=1,norm=False,lossfunc=F.mse_loss):
        super().__init__()
        self.scale=scaling
        self.normalize=norm
        self.loss=lossfunc
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1],
                                                      [-1, 0, 1],
                                                      [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', torch.tensor([[-1, -1, -1],
                                                      [0, 0, 0],
                                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, input, recon):
        # Gradient loss
        grad_input_x = F.conv2d(input, self.sobel_x, padding=1)
        grad_input_y = F.conv2d(input, self.sobel_y, padding=1)
        grad_recon_x = F.conv2d(recon, self.sobel_x, padding=1)
        grad_recon_y = F.conv2d(recon, self.sobel_y, padding=1)

        if self.normalize:
            input=normalize_tensor(input)
            recon=normalize_tensor(recon)

            grad_recon_x=normalize_tensor(grad_recon_x)
            grad_recon_y=normalize_tensor(grad_recon_y)
            grad_input_x=normalize_tensor(grad_input_x)
            grad_input_y=normalize_tensor(grad_input_y)

        pixel_loss = self.loss(recon, input,reduction="sum")
        grad_loss = self.loss(grad_input_x, grad_recon_x,reduction="sum")/2 + self.loss(grad_input_y, grad_recon_y,reduction="sum")/2

        return pixel_loss + self.scale * grad_loss
# Simple VAE class
class VAE(nn.Module):
    def __init__(self,inputshape,latent_dim,convchannels=[16,32],fc_layers=[512,256],beta=0.1,imageloss=DiffImageLoss(0.5,True)):
        super(VAE, self).__init__()
        self.beta=beta
        self.image_loss=imageloss
        self.conv_dim = (
        convchannels[-1], inputshape[0] // (2 ** len(convchannels)), inputshape[1] // (2 ** len(convchannels)))
        #Loop over convchannels and append conv maxpool/conv upscale to lists
        convchannels.insert(0,1)
        down=[]

        up=[]
        for i in range(len(convchannels)-1):
            down.append(nn.Conv2d(convchannels[i],convchannels[i+1],kernel_size=(3,3),padding=1))
            down.append(nn.ReLU())
            down.append(nn.MaxPool2d(kernel_size=2, stride=2))

            if not i==0:
                up.append(nn.ReLU())
            else:
                up.append(nn.Sigmoid())
            up.append(nn.Conv2d(convchannels[i+1],convchannels[i],kernel_size=(3,3),padding=1))
            up.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))


        down.append(torch.nn.Flatten(start_dim=1))

        up.append(ReshapeLayer(*self.conv_dim))
        up.append(nn.ReLU())
        up.append(nn.Linear(fc_layers[0],self.conv_dim[0]*self.conv_dim[1]*self.conv_dim[2] ))
        up.reverse()

        #Loop over fc_layers and add to list
        down_linear=[]
        up_linear=[]
        for i in range(len(fc_layers)):
            down_linear.append(nn.LazyLinear(fc_layers[i]))
            down_linear.append(nn.ReLU())

            up_linear.append(nn.ReLU())
            up_linear.append(nn.LazyLinear(fc_layers[i]))

        #up_linear.append(nn.Linear(latent_dim,fc_layers[-1]))
        up_linear.reverse()

        self.mu=nn.Linear(fc_layers[-1],latent_dim)
        self.logvar=nn.Linear(fc_layers[-1],latent_dim)

        #Turn into sequentials for decode and encode
        self.down=nn.Sequential(*down,*down_linear)
        self.decode=nn.Sequential(*up_linear,*up)

        dummy=torch.zeros(1,1,*inputshape)
        self.forward(dummy)


    def encode(self, x):
        h=self.down(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_loss(self,recon_x, x,logvar,mu,angle=0):
        #Perform inverse rotation to judge reconstruction in fixed reference direction
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_x = rotate(recon_x, -angle.squeeze())
        return  self.image_loss(recon_x,x) +self.beta * KLD

def randomrot(batch):
    return rotate(batch,torch.rand(batch.shape[0],device=batch.device)*180-90)

class ME_VAE(nn.Module):
    def __init__(self, inputshape, latent_dim, convchannels=[16, 32], fc_layers=[512, 256],num_encoders=3,transform=randomrot, beta=1,
                 imageloss=DiffImageLoss(0.5, True)):
        super(ME_VAE, self).__init__()
        #setting a callable transform function
        self.transform=transform

        self.n_enc=num_encoders
        self.beta = beta
        #setting a callable imageloss
        self.image_loss = imageloss

        #the dimension of the input for the convolutional part of the decoder.
        self.conv_dim = (
            convchannels[-1], inputshape[0] // (2 ** len(convchannels)), inputshape[1] // (2 ** len(convchannels)))

        # Loop over convchannels and append conv maxpool/conv upscale to lists
        convchannels.insert(0, 1)
        down = []
        for i in range(self.n_enc):
            down.append([])

        up = []
        for i in range(len(convchannels) - 1):
            for j in range(self.n_enc):
                down[j].append(nn.Conv2d(convchannels[i], convchannels[i + 1], kernel_size=(3, 3), padding=1))
                down[j].append(nn.ReLU())
                down[j].append(nn.MaxPool2d(kernel_size=2, stride=2))

            if not i == 0:
                up.append(nn.ReLU())
            else:
                up.append(nn.Sigmoid())
            up.append(nn.Conv2d(convchannels[i + 1], convchannels[i], kernel_size=(3, 3), padding=1))
            up.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        for i in range(self.n_enc):
            down[i].append(torch.nn.Flatten(start_dim=1))

        up.append(ReshapeLayer(*self.conv_dim))
        up.append(nn.ReLU())
        up.append(nn.Linear(fc_layers[0], self.conv_dim[0] * self.conv_dim[1] * self.conv_dim[2]))
        up.reverse()

        # Loop over fc_layers and add to list
        down_linear = []
        for i in range(self.n_enc):
            down_linear.append([])

        up_linear = []
        for i in range(len(fc_layers)):
            for j in range(self.n_enc):
                down_linear[j].append(nn.LazyLinear(fc_layers[i]))
                down_linear[j].append(nn.ReLU())

            up_linear.append(nn.ReLU())
            up_linear.append(nn.LazyLinear(fc_layers[i]))



        up_linear.reverse()

        self.mu = nn.ModuleList([
            nn.Linear(fc_layers[-1], latent_dim) for _ in range(self.n_enc)
        ])
        self.logvar = nn.ModuleList([
            nn.Linear(fc_layers[-1], latent_dim) for _ in range(self.n_enc)
        ])

        # Turn into sequentials for decode and encode

        self.down = nn.ModuleList([
            nn.Sequential(*down[i], *down_linear[i]) for i in range(self.n_enc)
        ])

        self.decode = nn.Sequential(*up_linear, *up)

        #initialize Lazy layers
        dummy = torch.zeros(1, 1, *inputshape)
        self.forward(dummy)

    def encode(self, x):
        h=[]
        mu=[]
        logvar=[]

        for i in range(self.n_enc):
            h.append(self.down[i](self.transform(x)))
            mu.append(self.mu[i](h[i]))
            logvar.append(self.logvar[i](h[i]))
        return torch.stack(mu),torch.stack(logvar)

    def reparameterize(self, mu, logvar):
        #mu logvar are tensors enc,batch,latentdim
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return torch.prod(mu + eps * std,axis=0)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_loss(self, recon_x, x, logvar, mu, angle=0):
        #perform inverse rotation to judge reconstruction in fixed reference direction
        recon_x = rotate(recon_x, -angle.squeeze())
        #take mean over list of mu,logvar for KLD loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/self.n_enc
        return self.image_loss(recon_x, x) + self.beta * KLD


class LinConv(nn.Module):
    def __init__(self, outputshape,inputsize, conv=[32, 16], lin=[256,512], final_act_sig=False):
        super(LinConv, self).__init__()


        #Computing correct scaling and padding
        initial_conv_h=outputshape[1]//(2**len(conv))
        pad_h=(outputshape[1]%(2**len(conv)))//len(conv)
        leftoverpad_h=(outputshape[1]%(2**len(conv)))%len(conv)

        initial_conv_w = outputshape[2] // (2 ** len(conv))
        pad_w = (outputshape[2] % (2 ** len(conv))) // len(conv)
        leftoverpad_w = (outputshape[2] % (2 ** len(conv))) % len(conv)

        #computing initial dim for input to conv
        self.conv_dim=(conv[0],initial_conv_h,initial_conv_w)
        #initialize linear layers
        self.linear=[]

        if not lin[0]==inputsize:#ensuring correct input dim
            lin.insert(0,inputsize)
        if not lin[-1]==conv[0]*initial_conv_h*initial_conv_w:#ensuring correct output dim
            lin.append(conv[0]*initial_conv_h*initial_conv_w)
        for i in range(len(lin)-1):
            self.linear.append(nn.Linear(lin[i],lin[i+1]))
            self.linear.append(nn.ReLU())
        self.linear=nn.Sequential(*self.linear)

        #initialize reshape function
        self.reshape=ReshapeLayer(*self.conv_dim)
        #initialize convolutional layers
        self.convolution=[]

        #ensuring correct output cahnnels
        conv.append(outputshape[0])
        for i in range(len(conv)-1):

            if i==0:
                #x = F.pad(x, (1, 1, 1, 0))
                self.convolution.append(nn.ConstantPad2d((pad_w//2+pad_w%2+leftoverpad_w//2+leftoverpad_w%2,pad_w//2+leftoverpad_w//2,
                                         pad_h//2+leftoverpad_h//2,pad_h//2+pad_h%2+leftoverpad_h//2+leftoverpad_h%2),0.0))
            else:
                self.convolution.append(nn.ConstantPad2d((pad_w // 2 + pad_w % 2, pad_w // 2, pad_h // 2, pad_h // 2 + pad_h % 2),0.0))
            self.convolution.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.convolution.append(nn.Conv2d(conv[i], conv[i + 1], kernel_size=(3, 3), padding=(1, 1)))
            if i==len(conv)-2 and final_act_sig:
                self.convolution.append(nn.Sigmoid())
            else:
                self.convolution.append(nn.ReLU())

        self.convolution=nn.Sequential(*self.convolution)

    def forward(self, x):
        x = self.linear(x)
        x = self.reshape(x)
        return self.convolution(x)

class ConvLin(nn.Module):
    def __init__(self,inputshape,conv=[16,32],lin=[512,256],final_act_sig=False):
        super(ConvLin, self).__init__()
        if not conv[0]==inputshape[0]:
            conv.insert(0,inputshape[0])
        self.conv=[]
        for i in range(len(conv)-1):
            self.conv.append(nn.Conv2d(conv[i], conv[i + 1], kernel_size=(3, 3), padding=1))
            self.conv.append(nn.ReLU())
            self.conv.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv=torch.nn.Sequential(*self.conv)
        self.lin=[]
        with torch.no_grad():
            dummy=torch.zeros(1,*inputshape)
            enc_shape=self.conv(dummy).shape

        lin.insert(0,enc_shape[1]*enc_shape[2]*enc_shape[3])

        for i in range(len(lin)-1):
            self.lin.append(nn.Linear(lin[i],lin[i+1]))
            if final_act_sig and i==len(lin)-2:
                self.lin.append(nn.Sigmoid)
            else:
                self.lin.append(nn.ReLU())
        self.lin=torch.nn.Sequential(*self.lin)

    def forward(self,x):
        x=self.conv(x)
        x=torch.flatten(x,start_dim=-3)
        return self.lin(x)

class VAE_modular(nn.Module):
    def __init__(self, inputshape, latent_dim, down, decoder, beta=1.0, imageloss=DiffImageLoss(0.5, True)):
        super(VAE, self).__init__()
        self.beta = beta
        self.image_loss = imageloss

        self.down = down
        self.decoder = decoder
        with torch.no_grad():
            dummy = torch.zeros((1, *inputshape))
            dummy_enc = self.down(dummy)  # should be batch,n

        self.mu = nn.Linear(dummy_enc.shape[1], latent_dim)
        self.logvar = nn.Linear(dummy_enc.shape[1], latent_dim)

    def encode(self, x):
        h = self.down(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_loss(self, recon_x, x, logvar, mu, angle=0):
        # Perform inverse rotation to judge reconstruction in fixed reference direction
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_x = rotate(recon_x, -angle.squeeze())
        return self.image_loss(recon_x, x) + self.beta * KLD

class ConvDown(nn.Module):
    def __init__(self,inputchannels,outputchannels,channels=[16,32],maxpool=True,batchnorm=False,doubleconv=False):
        super(ConvDown, self).__init__()
        self.conv=[]
        if not channels[0]==inputchannels:
            channels.insert(0,inputchannels)
        if not channels[-1]==outputchannels:
            channels.append(outputchannels)
        for i in range(len(channels)-1):
            self.conv.append(nn.Conv2d(channels[i],channels[i+1],kernel_size=(3,3),padding=(1,1)))
            if doubleconv:
                if batchnorm:
                    self.conv.append(nn.BatchNorm2d(channels[i+1]))
                self.conv.append(nn.GELU())
                self.conv.append(nn.Conv2d(channels[i+1], channels[i + 1], kernel_size=(3, 3), padding=(1, 1)))
            if batchnorm:
                self.conv.append(nn.BatchNorm2d(channels[i + 1]))
            self.conv.append(nn.GELU())
            if maxpool:
                self.conv.append(nn.MaxPool2d((2,2),2))
            else:
                nn.Conv2d(i+1,i+1, kernel_size=(4,4), stride=(2,2), padding=1)

        self.conv=nn.Sequential(*self.conv)
        print(self.conv)
    def forward(self,inputs):
        return self.conv(inputs)

class ConvUp(nn.Module):
    def __init__(self,inputchannels,outputchannels,channels=[32,16],doubleconv=False,last_act_sig=False):
        super(ConvUp, self).__init__()
        self.conv=[]
        if not channels[0]==inputchannels:
            channels.insert(0,inputchannels)
        if not channels[-1]==outputchannels:
            channels.append(outputchannels)
        for i in range(len(channels)-1):
            self.conv.append(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False))
            if doubleconv:
                self.conv.append(nn.Conv2d(channels[i], channels[i], kernel_size=(3, 3), padding=(1, 1)))
                self.conv.append(nn.GELU())
            self.conv.append(nn.Conv2d(channels[i],channels[i+1],kernel_size=(3,3),padding=(1,1)))


            if last_act_sig and i==len(channels)-2:
                self.conv.append(nn.Sigmoid())
            else:
                self.conv.append(nn.GELU())


        self.conv=nn.Sequential(*self.conv)
    def forward(self,inputs):
        return self.conv(inputs)


class VQLookUpTable(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost,refresh_every=-1,usage_threshold=1):
        super().__init__()
        #create lookup table
        self.n_emb=num_embeddings
        self.emb_dim=embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        #initialize with uniform embeddings
        #self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        #initialize with xavier uniform
        #nn.init.xavier_uniform_(self.embedding.weight)

        nn.init.normal_(self.embedding.weight,std=1.0)

        # 2) Optionally scale down (e.g. to σ=0.1)
        #self.embedding.weight.data.mul_(0.1)


        self.refresh_timer=refresh_every
        self.counter=0
        self.usage_threshold=usage_threshold
        self.refresh=(refresh_every>0)

        self.commitment_cost = commitment_cost
        self.register_buffer('usage', torch.zeros(self.n_emb, dtype=torch.long))



    def embedding_indices(self,inputs):
        # inputs: (B, D, H, W)
        # Flatten input to shape (B*H*W, D)
        B, D, H, W = inputs.shape

        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # Compute L2 distance between encoder outputs and embedding weights
        # dist: (B*H*W, num_embeddings)
        distances = (
                flat_input.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat_input @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=1)
        )
        # Find nearest embedding index for each input
        encoding_indices = torch.argmin(distances, dim=1)
        #unflattening
        encoding_indices = encoding_indices.view(B, H, W).unsqueeze(1)
        return encoding_indices

    def forward(self, inputs):
        # inputs: (B, D, H, W)
        # Flatten input to shape (B*H*W, D)
        B, D, H, W = inputs.shape

        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, D)
        # Compute L2 distance between encoder outputs and embedding weights
        # dist: (B*H*W, num_embeddings)
        distances = (
            flat_input.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_input @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        # Find nearest embedding index for each input
        encoding_indices = torch.argmin(distances, dim=1)
        # Quantize: lookup embeddings

        self.usage+=torch.bincount(encoding_indices,minlength=self.n_emb)

        if self.refresh and self.counter>self.refresh_timer:
            self.refresh_codebook(flat_input)
        else:
            self.counter+=1


        quantized = self.embedding(encoding_indices)  # (B*H*W, D)

        # Reshape back to (B, D, H, W)
        quantized = quantized.view(B, H, W, D).permute(0, 3, 1, 2)

        #passing the inputs passes their gradients
        #the rest to make it the looked up value is passed with detach
        quantized_st = inputs + (quantized - inputs).detach()

        # ------------- LOSS TERMS -------------
        # Codebook (embedding) loss: moves embeddings towards encoder outputs
        embed_loss = F.mse_loss(quantized, inputs.detach())
        # Commitment loss: encourages encoder outputs to stay close to embeddings
        commit_loss = self.commitment_cost * F.mse_loss(inputs, quantized.detach())
        vq_loss = embed_loss + commit_loss

        return quantized_st, vq_loss

    @torch.no_grad()
    def refresh_codebook(self,flat_input):
        device = self.embedding.weight.device
        #used_counts = torch.bincount(encoding_indices, minlength=self.n_emb)

        #Identify unused or low-usage codebook indices
        underused = (self.usage <= self.usage_threshold).nonzero(as_tuple=False).squeeze()

        if underused.numel() == 0:
            return  # all entries are fine

        # Select random input vectors to replace them with
        rand_input_indices = torch.randint(0, flat_input.shape[0], (underused.shape[0],), device=device)
        replacement_vectors = flat_input[rand_input_indices]

        # Replace the dead entries
        self.embedding.weight.data[underused] = replacement_vectors
        self.usage*=0
        self.counter=0



class VQ_VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 input_shape: tuple[int,int,int],
                 num_embeddings: int = 512,
                 commitment_cost: float = 0.25,imageloss: nn.Module=DiffImageLoss(),
                 codebook_refresh_period: int=-1,codebook_usage_threshold=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.img_loss=imageloss
        dummy = torch.zeros((1, *input_shape))

        with torch.no_grad():
            out = self.encoder(dummy)

        _, D, Hq, Wq = out.shape

        #build a lookuptable that maps D-dim vectors
        #at each of the Hq×Wq positions into a codebook of size num_embeddings
        self.vq = VQLookUpTable(
            num_embeddings=num_embeddings,
            embedding_dim=D,
            commitment_cost=commitment_cost,
            refresh_every=codebook_refresh_period,
            usage_threshold=codebook_usage_threshold
        )

    def get_indices(self,x):
        z_e=self.encoder(x)
        ind=self.vq.embedding_indices(z_e)
        return ind

    def forward(self, x):
        z_e = self.encoder(x)
        # Quantize and compute VQ loss
        z_q, vq_loss = self.vq(z_e)
        # Reconstruct
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss
    def get_loss(self,x,x_recon,vq_loss):
        return self.img_loss(x,x_recon)+vq_loss


def train_vq_vae(model, dataloader, optimizer, epochs, device='cuda' if torch.cuda.is_available() else 'cpu',
              save_path='vq_vae_checkpoint'):
    model=model.to(device)

    try:
        for epoch in range(1,epochs+1):
            model.train()
            total_loss=0
            progress_bar=tqdm(dataloader,desc=f"Epoch {epoch}/{epochs}")
            for images,angles in progress_bar:
                images=images.to(device)
                angles=angles.to(device)

                optimizer.zero_grad()

                recon_x,vq_loss=model(images)

                recon_x=rotate(recon_x,-angles.squeeze())

                loss=model.get_loss(images,recon_x,vq_loss)
                loss.backward()

                optimizer.step()

                total_loss+=loss.item()
                progress_bar.set_postfix(loss=loss.item())
            avg_loss=total_loss/len(dataloader.dataset)
            print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def train_vae(model, dataloader, optimizer, epochs, device='cuda' if torch.cuda.is_available() else 'cpu',
              save_path='small_cell_VAE.pt'):
    model = model.to(device)

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

            for images, angles in progress_bar:
                images = images.to(device)
                angles = angles.to(device)

                optimizer.zero_grad()
                recon_x, mu, logvar = model(images)
                loss = model.get_loss(recon_x, images, logvar, mu, angle=angles)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")




