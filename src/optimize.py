import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import librosa
import numpy as np
from tqdm import tqdm
from datetime import datetime

####### TEMPORARY ########
flnA = 'bongo-loop.mp3'
flnB = 'crickets.mp3'

args = {
    'audio_path': '../data/',
    'sr': 22050,
    'nfft': 2048,
    'hoplen': 1024,
    'init_downbeat': False,  # whether to set first detected beat to first downbeat
    'target_pattern': 'B'  # target output beat length from files
}

print('loading files...')
inputA, _ = librosa.load(args['audio_path'] + flnA, sr=args['sr'], mono=True)
inputB, _ = librosa.load(args['audio_path'] + flnB, sr=args['sr'], mono=True)

# same size
length = min(len(inputA), len(inputB))
inputA = inputA[:length]
inputB = inputB[:length]

n = 2048
eps = 1E-8
# (the default params of librosa.stft is consistent w/ tomczak et al)
A = np.log(np.abs(librosa.stft(inputA)) + eps)  # SHAPE (1 + n/2, T)
B = np.log(np.abs(librosa.stft(inputB)) + eps)
_, T = A.shape  # A.shape == B.shape because of the slicing above ;)

# reshape from (n/2+1, T) to (1, n/2+1, T, 1) in Pytorch version i.e. (batch sz, channels, W, H)
# n/2+1 = depth/nb of channels
A = torch.from_numpy(np.ascontiguousarray(A[None, :, :, None])).float()  # CONTENT
B = torch.from_numpy(np.ascontiguousarray(B[None, :, :, None])).float()  # STYLE
Y = torch.from_numpy(np.random.rand(*A.shape)).float()
print('T=', T, 'n=', n)
print(A.shape)
##########################
# MODEL
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.cnn = nn.Conv2d(in_channels=n//2+1, out_channels=2*n, kernel_size=(16,1))
        self.selu = nn.SELU()
    def forward(self, A, B, Y):
        # CONTENT (shape (None, 287, 4096))
        a = self.selu(self.cnn(A)).squeeze(-1)
        b = self.selu(self.cnn(B)).squeeze(-1)
        y = self.selu(self.cnn(Y)).squeeze(-1)
        # STYLE (shape (None, 4096, 4096))
        g_b = torch.matmul(b, torch.transpose(b, 1, 2))
        g_y = torch.matmul(y, torch.transpose(y, 1, 2))
        return a, g_b, y, g_y
model = NeuralNetwork()
########################
# OPTIMIZATION
criterion_content = torch.nn.MSELoss(reduce='sum')
criterion_style = torch.nn.MSELoss(reduce='sum')
Y.requires_grad = True
optimizer = torch.optim.Adam([Y], lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

for iter in range(50):
    print("Epoch", iter)
    ## forward:
    a, g_b, y, g_y = model(A, B, Y)
    # normalization const for style Gram:
    Q = g_b.shape[1] * g_b.shape[2]  # Q=NM, but M=g_br.shape[2] is included in torch.nn.MSELoss with reduce='average'
    loss_content = 2 * criterion_content(a, y)
    loss_style = 200 * criterion_style(g_b, g_y) / (Q * Q)
    loss = loss_content + loss_style
    print("content loss", loss_content)
    print('style loss', loss_style)
    print("Loss:", loss)
    ## backward:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # update step
    scheduler.step(loss)

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('../trained_models/model_scripted.pt') # Save
# model.load_state_dict(torch.load(PATH))
# model.eval()
