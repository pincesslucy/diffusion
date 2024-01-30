import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from unet import UNet
from dataset import Dataset

T = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

beta = (torch.linspace(1e-4, 0.02, T)).to(device)
alpha = (1. - beta).to(device)
alpha_bar = torch.cumprod(alpha, dim=0).to(device) #(1000,)
alpha_bar_pre = F.pad(alpha_bar[:-1], (1,0), value=1.).to(device) #(1001,)

#training
sqrt_alpha_bar = torch.sqrt(alpha_bar).to(device)
sqrt_one_minus_alpha_bar = torch.sqrt(1.-alpha_bar).to(device)

#sampling
recip_sqrt_alpha = torch.sqrt(1./alpha).to(device)
recip_sqrt_one_minus_alpha = torch.sqrt(1./(1.-alpha)).to(device)
epsilon_coef = ((1. - alpha) / sqrt_alpha_bar).to(device)
sigma = (beta * (1. - alpha_bar_pre)/1. - alpha_bar).to(device)

#sampling2
recip_sqrt_alpha = torch.sqrt(1./alpha_bar).to(device)
recip_sqrt_alpha_minus_one = torch.sqrt(1. / alpha_bar - 1.).to(device)
coef1 = torch.sqrt(alpha_bar_pre) * beta / (1. - alpha_bar).to(device)
coef2 = torch.sqrt(alpha) * (1. - alpha_bar_pre) / (1. - alpha_bar).to(device)
sigma = (beta * (1. - alpha_bar_pre)/(1. - alpha_bar)).to(device)

def select_alpha(alpha, t, x):
    B, *dims = x.shape
    alpha = torch.gather(alpha, index=t, dim=0)
    return alpha.view([B] + [1]*len(dims))

def train (model,x_0):
    t = torch.randint(T,size=(x_0.shape[0], )).to(device)
    epsilon = torch.randn_like(x_0).to(device)
    x_t = select_alpha(sqrt_alpha_bar, t, x_0) * x_0 + \
        select_alpha(sqrt_one_minus_alpha_bar, t, x_0) * epsilon
    loss = F.mse_loss(model(x_t, t), epsilon)
    return loss

def sample(model, x_T):
    x_t = x_T
    for time_step in reversed(range(T)):
        t = torch.full((x_T.shape[0], ), time_step, dtype=torch.long).to(device)
        eps = model(x_t, t)


        x0_predicted = select_alpha(recip_sqrt_alpha, t, eps) * x_t - \
            select_alpha(recip_sqrt_alpha, t, eps) * select_alpha(epsilon_coef, t, eps) * eps
        
        z = torch.randn_like(x_t) if time_step else 0
        var = torch.sqrt(select_alpha(sigma, t, eps)) * z
        
        x_t = x0_predicted + var
    x_0 = x_t
    
    return x_0

def sample2(model, x_T):
    x_t = x_T
    for time_step in reversed(range(T)):
        t = torch.full((x_T.shape[0], ), time_step, dtype=torch.long).to(device)
        eps = model(x_t, t)


        x0_predicted = select_alpha(recip_sqrt_alpha, t, eps) * x_t - \
            select_alpha(recip_sqrt_alpha_minus_one, t, eps) * eps
        
        mean = select_alpha(coef1, t, eps) * x0_predicted + \
            select_alpha(coef2, t, eps) * x_t

        z = torch.randn_like(x_t) if time_step else 0
        var = torch.sqrt(select_alpha(sigma, t, eps)) * z
        
        x_t = mean + var
        # print(x_t[0])
        # if np.isnan(x_t[0].detach().cpu().numpy()).any():
        #     print("Nan")
        #     break

        # 50t마다 이미지 출력
        # if time_step % 50 == 0:
        #     x_0 = x_t.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy() * 255
        #     plt.imshow(x_0[0].astype(np.uint8))
        #     plt.axis('off')  
        #     plt.show()
    x_0 = x_t
    
    return x_0

dataset = Dataset('./data/archive/img_align_celeba/img_align_celeba')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = UNet(T=T, ch=128, ch_mult=[1, 2, 2, 1], attn=[1],
             num_res_blocks=2, dropout=0.1).to(device)

optim = torch.optim.Adam(model.parameters(), lr=2e-4)

for e in range(1, 100+1):
    model.train()
    for i, x in enumerate(tqdm(iter(dataloader)), 1):
        optim.zero_grad()
        x = x.to(device)
        loss = train(model, x)
        loss.backward()
        optim.step()
        print("\r[Epoch: {} , Iter: {}/{}]  Loss: {:.3f}".format(e, i, len(dataloader), loss.item()), end='')
    print("\n> Eval at epoch {}".format(e))
    
    #save model
    torch.save(model.state_dict(), './save/epoch_{}.pth'.format(e))

    model.eval()
    with torch.no_grad():
        x_T = torch.randn(5, 3, 128, 128).to(device)
        x_0 = sample2(model, x_T)
        x_0 = x_0.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy() * 255
        for i in range(5):
            plt.imshow(x_0[i].astype(np.uint8))
            plt.axis('off')  
            plt.show()