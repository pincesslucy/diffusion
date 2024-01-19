import torch
from unet import UNet
from diffusion import sample2
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T = 1000
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