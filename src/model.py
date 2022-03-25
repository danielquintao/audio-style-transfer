import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import librosa
import numpy as np
from tqdm import tqdm

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
A = np.log(librosa.stft(inputA) + eps)  # SHAPE (1 + n/2, T)
B = np.log(librosa.stft(inputB) + eps)
_, T = A.shape  # A.shape == B.shape because of the slicing above ;)

# reshape from (n/2+1, T) to (1, n/2+1, T, 1) in Pytorch version i.e. (batch sz, channels, W, H)
# n/2+1 = depth/nb of channels
A_real = torch.from_numpy(np.ascontiguousarray(np.real(A[None, :, :, None]))).float()  # CONTENT
A_imag = torch.from_numpy(np.ascontiguousarray(np.imag(A[None, :, :, None]))).float()
B_real = torch.from_numpy(np.ascontiguousarray(np.real(B[None, :, :, None]))).float()  # STYLE
B_imag = torch.from_numpy(np.ascontiguousarray(np.imag(B[None, :, :, None]))).float()
Y_real = torch.from_numpy(np.random.rand(*A_real.shape)).float()
Y_imag = torch.from_numpy(np.random.rand(*A_real.shape)).float()
print('T=', T, 'n=', n)
print(A_real.shape)
##########################
# MODEL
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.cnn = nn.Conv2d(in_channels=n//2+1, out_channels=2*n, kernel_size=(16,1))
        self.selu = nn.SELU()
    def forward(self, A_real, A_imag, B_real, B_imag, Y_real, Y_imag):
        # CONTENT (shape (None, 287, 4096))
        ar = self.selu(self.cnn(A_real)).squeeze(-1)
        ai = self.selu(self.cnn(A_imag)).squeeze(-1)
        br = self.selu(self.cnn(B_real)).squeeze(-1)
        bi = self.selu(self.cnn(B_imag)).squeeze(-1)
        yr = self.selu(self.cnn(Y_real)).squeeze(-1)
        yi = self.selu(self.cnn(Y_imag)).squeeze(-1)
        # STYLE (shape (None, 4096, 4096))
        g_br = torch.matmul(br, torch.transpose(br, 1, 2))
        g_bi = torch.matmul(bi, torch.transpose(bi, 1, 2))
        g_yr = torch.matmul(yr, torch.transpose(yr, 1, 2))
        g_yi = torch.matmul(yi, torch.transpose(yi, 1, 2))
        return ar, ai, g_br, g_bi, yr, yi, g_yr, g_yi
model = NeuralNetwork()
########################
# OPTIMIZATION
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for iter in tqdm(range(10)):
    ar, ai, g_br, g_bi, yr, yi, g_yr, g_yi = model(A_real, A_imag, B_real, B_imag, Y_real, Y_imag)
    # normalization const for style Gram:
    Q = 2 * g_br.shape[1]  # Q=2NM, but M=g_br.shape[2] is included in torch.nn.MSELoss with reduce='average'
    loss = 0.5 * criterion(ar, yr, reduce='sum')

