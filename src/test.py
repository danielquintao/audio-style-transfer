import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import librosa
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

with open('config/vars.yml') as f:
    VARS = yaml.load(f, yaml.Loader)

args = VARS['args']

# load prediction
Y_stft = np.exp(np.load(VARS['outputs_path'] + 'log_mag_spectro_pachelbelbongo-loop.npy').squeeze()) - 1
# CORRECTION IN CASE WE TRUNCATED THE STFT BEFORE OPTIMIZATION (IMPORTANT):
n = 2048
if Y_stft.shape[0] < n / 2 + 1:
    Y_temp = Y_stft.copy()
    Y_stft = np.zeros((n // 2 + 1, Y_stft.shape[1]))
    Y_stft[:Y_temp.shape[0], :] = Y_temp

# load content and style
fln_content = VARS['flnA']
content, _ = librosa.load(args['audio_path'] + fln_content, sr=args['sr'], mono=True)
fln_style = VARS['flnB']
style, _ = librosa.load(args['audio_path'] + fln_style, sr=args['sr'], mono=True)

# fix phase of Y using phase of content
content_stft = librosa.stft(content)
exp_phase = np.exp(1j * np.angle(content_stft))
Tmin = min(exp_phase.shape[1], Y_stft.shape[1])
Y_stft = Y_stft[:,:Tmin].astype('complex')
Y_stft *= exp_phase[:, :Tmin]


Y_audio = librosa.core.istft(Y_stft)
Y_audio[:50] = 0
Y_audio[-50:] = 0
print(Y_audio.shape)

# change amplitude to save to wav
Y_audio /= np.max(Y_audio)
Y_audio *= np.iinfo(np.int16).max
write(VARS['audio_results_path'] + 'result_' + fln_content[:-4] + fln_style[:-4] + '.wav', args['sr'], Y_audio.astype(np.int16))

fig, ax = plt.subplots(1, 3)
ax[0].plot(content)
ax[0].set_title('content')
ax[1].plot(style)
ax[1].set_title('style')
ax[2].plot(Y_audio)
ax[2].set_title('mixture')
plt.show()

#====================================

# output = output.squeeze(0)
# output = output.numpy()
# # print(output.shape)
# # output = output.resize([1025,2500])
#
# N_FFT = VARS['n'] # 2048
# a = np.zeros_like(output)
# a = np.exp(output) - 1
#
# # This code is supposed to do phase reconstruction
# p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
# for i in range(500):
#     S = a * np.exp(1j * p)
#     x = librosa.istft(S)
#     p = np.angle(librosa.stft(x, N_FFT))
