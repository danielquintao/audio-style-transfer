import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import librosa
import numpy as np

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
# CONTENT
cnn = nn.Conv2d(in_channels=n//2+1, out_channels=2*n, kernel_size=(16,1))
selu = nn.SELU()
ar = selu(cnn(A_real))
ai = selu(cnn(A_imag))
br = selu(cnn(B_real))
bi = selu(cnn(B_imag))
yr = selu(cnn(Y_real))
yi = selu(cnn(Y_imag))
print(ar.shape)
#########################
# COMPUTATION OF GRAM MATRIX
g_br = torch.mm(torch.squeeze(br), torch.transpose(torch.squeeze(br)))
g_bi = torch.mm(torch.squeeze(bi), torch.transpose(torch.squeeze(bi)))
g_yr = torch.mm(torch.squeeze(yr), torch.transpose(torch.squeeze(yr)))
g_yi = torch.mm(torch.squeeze(yi), torch.transpose(torch.squeeze(yi)))
print(g_br.shape)
########################
# OPTIMIZATION



