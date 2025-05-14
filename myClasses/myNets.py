import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import translate,rotate
from tqdm import tqdm

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
    def __init__(self,model,n_transforms=8,**kwargs):
        super(Localizer, self).__init__()
        self.model = model
        self.n_transforms=n_transforms
        return
    def forward(self,x):
        return self.model(x)

    def forward_tranform(self,batch,translation,angles):
        transformed=image_translation(batch,translation)
        return image_rotation(transformed,angles)

    def inverse_tranform(self,pred,translation,angles):
        invpred=inverse_rotation(pred,angles)
        return inverse_translation(invpred,translation)

    def get_loss(self,image):

        #THIS ALL NEEDS TO BE REWRITTEN

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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
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

        self.commitment_cost = commitment_cost


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
        '''
        with torch.no_grad():
            unique_codes = torch.unique(encoding_indices)
            print(unique_codes)
            print(f"Codebook usage: {len(unique_codes)} / {self.embedding.num_embeddings}")
            print(f"flat_input stats: mean={flat_input.mean().item():.4f}, std={flat_input.std().item():.4f}")
            print(
                f"embedding stats: mean={self.embedding.weight.mean().item():.4f}, std={self.embedding.weight.std().item():.4f}")
            print(distances[0])  # should vary across embeddings
            print(torch.argmin(distances[0]))  # check if it's always 1
        '''
        return quantized_st, vq_loss


class VQ_VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 input_shape: tuple[int,int,int],
                 num_embeddings: int = 512,
                 commitment_cost: float = 0.25,imageloss: nn.Module=DiffImageLoss()):
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
            commitment_cost=commitment_cost
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




