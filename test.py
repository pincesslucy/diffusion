import torch
from unet import UNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import imageio
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T = 1000

beta = (torch.linspace(1e-4, 0.02, T)).to(device)
alpha = (1. - beta).to(device)
alpha_bar = torch.cumprod(alpha, dim=0).to(device) #(1000,)
alpha_bar_pre = F.pad(alpha_bar[:-1], (1,0), value=1.).to(device) #(1001,)

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
        if time_step % 50 == 0:
            x_0 = x_t.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy() * 255
            n = random.randint(0, 4)
            plt.imshow(x_0[2].astype(int))
            plt.axis('off')  
            plt.savefig(f"./save/gif_sample/image_{time_step}.png") 
            plt.close()

    folder_path = './save/gif_sample/'
    # 주어진 폴더에 있는 PNG 파일들의 리스트를 생성합니다.
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]
    image_files.sort(reverse=True)
    images = [imageio.imread(image_file) for image_file in image_files]
    imageio.mimsave(folder_path+'output.gif', images, fps=10)
    x_0 = x_t
    
    return x_0

model = UNet(T=T, ch=128, ch_mult=[1, 2, 2, 1], attn=[1],
             num_res_blocks=2, dropout=0.1)
model.load_state_dict(torch.load('./save/epoch_1.pth'))
model.to(device)

model.eval()
with torch.no_grad():
    x_T = torch.randn(5, 3, 32, 32).to(device)
    x_0 = sample2(model, x_T)
    x_0 = x_0.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy() * 255
    for i in range(5):
        plt.imshow(x_0[i].astype(np.uint8))
        plt.axis('off')  
        plt.show()