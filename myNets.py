import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import translate,rotate

class DoubleConv(nn.Module):
    """(Conv => ReLU => Conv => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels,scaling=2):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(scaling),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True,scaling=2):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scaling, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=scaling)
            self.conv = DoubleConv(in_channels, out_channels)

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
    def __init__(self, n_channels, n_classes, layers=[64, 128, 256, 512], bilinear=True, scaling=2):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.downs=nn.ModuleList()
        self.ups=nn.ModuleList()

        self.inc=DoubleConv(n_channels, layers[0])
        self.outc=nn.Sequential(nn.Conv2d(layers[0], n_classes, kernel_size=1),nn.ReLU())

        self.nlayers=len(layers)

        self.xlist=[None]*self.nlayers
        for i in range(self.nlayers-1):
            self.downs.append(Down(layers[i], layers[i + 1], scaling=scaling))
            self.ups.append(Up(layers[self.nlayers-1-i]+layers[self.nlayers-2-i],layers[self.nlayers-2-i],scaling=scaling))
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



def image_flip(batch,flipx):
    pass

def inverse_flip(preds,flipx):
    pass


class Localizer(nn.Module):
    def __init__(self,model,n_transforms=8,**kwargs):
        super(LocalizerClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model=nn.Sequential(model,torch.nn.Sigmoid()).to(self.device)
        self.model = model.to(self.device)
        #self.loss=LodestarLoss(beta)
        self.n_transforms=n_transforms


        return
    def forward(self,x):

        return self.model(x.to(self.device))


    def consistency_loss(self, invpred,probmap):
        # invpred: [B, T, 2]
        # probmap: [B,T,1,l,l]
        diffs = invpred[:, 1:, :] - invpred[:, :-1, :]  # [B, T-1, 2]
        mse_per_sample = torch.mean((diffs ** 2).sum(dim=-1), dim=1)  # [B]

        total_mass = probmap.sum(dim=[1, 2,3,4])  # [B]
        mass_loss = -torch.log(0.1*total_mass + 1e-8)

        return mse_per_sample + self.alpha * mass_loss  # [B]


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
        tr = torch.rand(b, self.n_transforms, 2, device=images.device) * h//6 - h//12  # translations
        ag = torch.rand(b, self.n_transforms, device=images.device) * 360  # angles
        #flatten random args
        tr_flat = tr.view(b * self.n_transforms, 2)
        ag_flat = ag.view(b * self.n_transforms, )

        #transform images, run model
        transform_im=self.forward_tranform(flat,tr_flat,ag_flat)
        pred_flat=self.model(transform_im)

        preds=pred_flat.view(b,self.n_transforms,1,h,w)

        centroids_flat=mass_centroid(pred_flat)


        #invert transforms (do some flattening shit ig)
        invpred=self.inverse_tranform(centroids_flat.squeeze(),tr_flat,ag_flat)
        invpred = invpred.view(b, self.n_transforms, 2)

        # invpred: [B, T, 2]

        diffs = invpred[:, 1:, :] - invpred[:, :-1, :]  # [B, T-1, 2]
        mse_per_sample = torch.mean((diffs ** 2).sum(dim=-1), dim=1)  # [B]
        return mse_per_sample





class LocalizerClassifier(nn.Module):
    def __init__(self,model,n_transforms=8,beta=0.1,alpha=1.5,**kwargs):
        super(LocalizerClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model=nn.Sequential(model,torch.nn.Sigmoid()).to(self.device)
        self.model = model.to(self.device)
        #self.loss=LodestarLoss(beta)
        self.n_transforms=n_transforms

        self.beta=beta
        self.alpha=alpha

        return
    def forward(self,x):

        return self.model(x.to(self.device))


    def consistency_loss(self, invpred,probmap):
        # invpred: [B, T, 2]
        # probmap: [B,T,1,l,l]
        diffs = invpred[:, 1:, :] - invpred[:, :-1, :]  # [B, T-1, 2]
        mse_per_sample = torch.mean((diffs ** 2).sum(dim=-1), dim=1)  # [B]

        total_mass = probmap.sum(dim=[1, 2,3,4])  # [B]
        mass_loss = -torch.log(0.1*total_mass + 1e-8)

        return mse_per_sample + self.alpha * mass_loss  # [B]


    def forward_tranform(self,batch,translation,angles):

        transformed=image_translation(batch,translation)
        return image_rotation(transformed,angles)

    def inverse_tranform(self,pred,translation,angles):
        invpred=inverse_rotation(pred,angles)
        return inverse_translation(invpred,translation)

    def get_loss(self,image,detect):
        b,c,h,w=image.shape

        detect=detect.to(self.device).float()

        #expanding for transform
        images=image.unsqueeze(1).expand(-1,self.n_transforms,-1,-1,-1).contiguous()

        #flattening to feed into network
        flat=images.view(b*self.n_transforms,c,h,w)

        #getting random args
        tr = torch.rand(b, self.n_transforms, 2, device=images.device) * h//6 - h//12  # translations
        ag = torch.rand(b, self.n_transforms, device=images.device) * 360  # angles
        #flatten random args
        tr_flat = tr.view(b * self.n_transforms, 2)
        ag_flat = ag.view(b * self.n_transforms, )

        #transform images, run model
        transform_im=self.forward_tranform(flat,tr_flat,ag_flat)
        pred_flat=self.model(transform_im)

        preds=pred_flat.view(b,self.n_transforms,1,h,w)

        centroids_flat=mass_centroid(pred_flat)


        #invert transforms (do some flattening shit ig)
        invpred=self.inverse_tranform(centroids_flat.squeeze(),tr_flat,ag_flat)
        invpred = invpred.view(b, self.n_transforms, 2)


        loss = torch.lerp(
            preds.sum(dim=[1,2,3,4]) * self.beta,  # if detect_mask==0
            self.consistency_loss(invpred,preds),  # if detect_mask==1
            detect  # interpolation factor: 0 or 1
        ).sum()

        return loss


def train_localizer(loc, dataloader, optimizer, epochs=300,filename="filename"):
    loc.train()
    device = loc.device
    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for inputs, detect in dataloader:
                inputs = inputs.to(device)  # [B, C, H, W]
                detect = detect.to(device)  # [B]

                optimizer.zero_grad()
                loss = loc.get_loss(inputs, detect)
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