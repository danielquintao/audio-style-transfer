import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

args = {  # TODO set global
    'audio_path': '../data/',
    'sr': 22050,
    'nfft': 2048,
    'hoplen': 1024,
    'init_downbeat': False,  # whether to set first detected beat to first downbeat
    'target_pattern': 'B'  # target output beat length from files
}

# load prediction
Y_stft = np.exp(np.load('../outputs/log_mag_spectro_pachelbelbongo-loop.npy').squeeze()) - 1

# load content and style
fln_content = 'pachelbel.mp3'
content, _ = librosa.load(args['audio_path'] + fln_content, sr=args['sr'], mono=True)
fln_style = 'bongo-loop.mp3'
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
write('../outputs/result_'+ fln_content[:-4] + fln_style[:-4] + '.wav', args['sr'], Y_audio.astype(np.int16))

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
# N_FFT = 2048
# a = np.zeros_like(output)
# a = np.exp(output) - 1
#
# # This code is supposed to do phase reconstruction
# p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
# for i in range(500):
#     S = a * np.exp(1j * p)
#     x = librosa.istft(S)
#     p = np.angle(librosa.stft(x, N_FFT))
