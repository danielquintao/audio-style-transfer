import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import librosa
import numpy as np
from tqdm import tqdm
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open('config/vars.yml') as f:
    VARS = yaml.load(f, yaml.Loader)

flnA = VARS['flnA']
flnB = VARS['flnB']

args = VARS['args']

print('loading files...')
inputA, _ = librosa.load(args['audio_path'] + flnA, sr=args['sr'], mono=True)
inputB, _ = librosa.load(args['audio_path'] + flnB, sr=args['sr'], mono=True)

# same size
length = min(len(inputA), len(inputB))
inputA = inputA[:length]
inputB = inputB[:length]

n = VARS['n']  # 2048
eps = VARS['eps']  # 1

# (the default params of librosa.stft is consistent w/ tomczak et al)
A = np.log(np.abs(librosa.stft(inputA)) + eps)[:n//4 + 1,:]  # SHAPE (1 + n/2, T) # XXX
B = np.log(np.abs(librosa.stft(inputB)) + eps)[:n//4 + 1,:]                       # XXX
# XXX REDEFINE n:
n = n // 2
# define T
_, T = A.shape  # A.shape == B.shape because of the slicing above ;)

# reshape from (n/2+1, T) to (1, n/2+1, T, 1) in Pytorch version i.e. (batch sz, channels, W, H)
# n/2+1 = depth/nb of channels
A = torch.from_numpy(np.ascontiguousarray(A[None, :, :, None])).float()  # CONTENT
B = torch.from_numpy(np.ascontiguousarray(B[None, :, :, None])).float()  # STYLE
Y = torch.from_numpy(np.random.rand(*A.shape) * 1E-3).float() # XXX
print('T=', T, 'n=', n)
print(A.shape)
##########################
# MODEL
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.cnn = nn.Conv2d(in_channels=n//2+1, out_channels=2*n, kernel_size=(16,1))
        torch.nn.init.normal_(self.cnn.weight, std=np.sqrt(2/(n)))  # original sqrt(2/n^3), found by hard specting source code
        self.selu = nn.SELU()
    def forward(self, A, B, Y):
        # CONTENT (shape (None, 287, 4096))
        a = self.selu(self.cnn(A.to(device))).squeeze(-1)
        b = self.selu(self.cnn(B.to(device))).squeeze(-1)
        y = self.selu(self.cnn(Y.to(device))).squeeze(-1)
        # STYLE (shape (None, 4096, 4096))
        g_b = torch.matmul(b, torch.transpose(b, 1, 2))
        g_y = torch.matmul(y, torch.transpose(y, 1, 2))
        Q = b.shape[1]  # XXX * b.shape[2]  # Q=NM
        g_b = torch.divide(g_b, Q)
        g_y = torch.divide(g_y, Q)
        return a, g_b, y, g_y

model = NeuralNetwork()
Y.requires_grad = True
########################
# GPU
model.to(device)
########################
# OPTIMIZATION
criterion_content = torch.nn.MSELoss(reduce='sum')
criterion_style = torch.nn.MSELoss(reduce='sum')
optimizer = torch.optim.AdamW([Y], lr=.5)  # XXX
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, verbose=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 7, threshold=0.001)
# fight against overfitting the content:
last_style_loss = np.inf
count_style_loss_augmen = 0

losses_style = []
losses_content = []
losses = []

for iter in range(500):
    print("Epoch", iter)
    ## forward:
    a, g_b, y, g_y = model(A, B, Y)
    # normalization const for style Gram:
    loss_content = 2 * criterion_content(a, y)
    loss_style = 2 * criterion_style(g_b, g_y) * 100
    loss = loss_content + loss_style
    print("content loss", loss_content)
    losses_content.append(loss_content.detach().cpu().numpy())
    print('style loss', loss_style)
    losses_style.append(loss_style.detach().cpu().numpy())
    print("Loss:", loss)
    losses.append(loss.detach().cpu().numpy())
    ## backward:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # update step
    scheduler.step(loss_style)  # XXX SCHEDULE USING ONLY ON THE LOSS_STYLE
    # early stop
    if loss_style > last_style_loss:
        count_style_loss_augmen += 1
    else:
        count_style_loss_augmen = 0
    if count_style_loss_augmen > 10:
        print('too many consecutive style losses achieved -> EARLY STOPPING')
        break  # early stop!
    last_style_loss = loss_style


np.save(VARS['outputs_path'] + VARS['flnLogMagSpectrogram'], Y.detach().cpu().numpy().squeeze())

