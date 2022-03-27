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
Y_logstft = np.load(VARS['outputs_path'] + 'log_mag_spectro_pachelbelbongo-loop.npy').squeeze()
Y_stft = np.exp(Y_logstft) - 1

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

# convert content and style to log1p(stft) for further comparison
content_logstft = np.log(np.abs(librosa.stft(content)) + 1)[:len(Y_logstft):]
style_logstft = np.log(np.abs(librosa.stft(style)) + 1)[:len(Y_logstft):]

# fix phase of Y using phase of content
content_stft = librosa.stft(content)
exp_phase = np.exp(1j * np.angle(content_stft))
Tmin = min(exp_phase.shape[1], Y_stft.shape[1])
Y_stft = Y_stft[:,:Tmin].astype('complex')
Y_stft *= exp_phase[:, :Tmin]


Y_audio = librosa.core.istft(Y_stft)
Y_audio[:750] = 0
Y_audio[-750:] = 0
print(Y_audio.shape)

# change amplitude to save to wav
Y_audio /= np.max(Y_audio)
Y_audio *= np.iinfo(np.int16).max
write(VARS['audio_results_path'] + 'result_' + fln_content[:-4] + fln_style[:-4] + '.wav', args['sr'], Y_audio.astype(np.int16))

# compare waveforms
fig, ax = plt.subplots(1, 3)
ax[0].plot(content)
ax[0].set_title('content')
ax[1].plot(style)
ax[1].set_title('style')
ax[2].plot(Y_audio)
ax[2].set_title('mixture')
plt.show()

# compare spectrograms
fig, ax = plt.subplots(1, 3)
ax[0].imshow(content_logstft)
ax[0].set_title('content')
ax[1].imshow(style_logstft)
ax[1].set_title('style')
ax[2].imshow(Y_logstft)
ax[2].set_title('mixture')
plt.show()

# evaluate content
# pitches
Tmin = min(len(Y_audio), len(content), len(style))
ccens_Y = librosa.feature.chroma_cens(Y_audio[:Tmin], sr=args['sr'], hop_length=args['hoplen'])
ccens_content = librosa.feature.chroma_cens(content[:Tmin], sr=args['sr'], hop_length=args['hoplen'])
ccens_style = librosa.feature.chroma_cens(style[:Tmin], sr=args['sr'], hop_length=args['hoplen'])
# cosine similarity between chroma cens
norm_Y = np.sqrt(np.sum(ccens_Y * ccens_Y))
norm_content = np.sqrt(np.sum(ccens_content * ccens_content))
norm_style = np.sqrt(np.sum(ccens_style * ccens_style))
sim_Y_content = np.dot(ccens_Y.flatten(), ccens_content.flatten()) / (norm_Y * norm_content)
sim_Y_style = np.dot(ccens_Y.flatten(), ccens_style.flatten()) / (norm_Y * norm_style)
sim_content_style = np.dot(ccens_content.flatten(), ccens_style.flatten()) / (norm_content * norm_style)  # expected to be small
print('COSINE SIMILARITIES BETWEEN CHROMA-CENS:')
print('             content   style     mixture')
print('content {:10.4f}{:10.4f}{:10.4f}'.format(1, sim_content_style, sim_Y_content))
print('style   {:10.4f}{:10.4f}{:10.4f}'.format(sim_content_style, 1, sim_Y_style))
print('mixture {:10.4f}{:10.4f}{:10.4f}'.format(sim_Y_content, sim_Y_style, 1))
# visualize CHROMA-CENS
fig, ax = plt.subplots(1, 3, figsize=(1,3))
ax[0].imshow(ccens_content)
ax[0].set_title('content')
ax[1].imshow(ccens_style)
ax[1].set_title('style')
ax[2].imshow(ccens_Y)
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
