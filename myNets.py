import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import translate,rotate
import tqdm
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


def train_localizer_classifier(loc, dataloader, optimizer, epochs=300, filename="filename"):
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


def train_localizer(loc, dataloader, optimizer, epochs=300, filename="filename"):
    loc.train()
    device = loc.device
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

# Simple VAE class
class VAE(nn.Module):
    def __init__(self,inputshape,latent_dim,convchannels=[16,32],fc_layers=[512,256],beta=0.1):
        super(VAE, self).__init__()
        self.beta=beta
        self.conv_dim = (
        convchannels[-1], inputshape[0] // (2 ** len(convchannels)), inputshape[1] // (2 ** len(convchannels)))
        #Loop over convchannels and append conv maxpool/conv upscale to lists
        convchannels.insert(0,1)
        down=[]

        up=[]
        for i in range(len(convchannels)-1):
            down.append(nn.Conv2d(convchannels[i],convchannels[i+1],kernel_size=(3,3)))
            down.append(nn.ReLU())
            down.append(nn.MaxPool2d(kernel_size=2, stride=2))

            if not i==0:
                up.append(nn.ReLU())
            else:
                up.append(nn.Sigmoid())
            up.append(nn.Conv2d(convchannels[i+1],convchannels[i],kernel_size=(3,3)))
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

        #print(self.down)
        #print(self.decode)

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
        recon_x=rotate(recon_x,-angle)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + self.beta*KLD


def train_vae(model, dataloader, optimizer, epochs, device='cuda' if torch.cuda.is_available() else 'cpu',
              save_path='vae_checkpoint.pt'):
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




