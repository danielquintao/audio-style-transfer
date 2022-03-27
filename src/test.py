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
Y_logstft = np.load(VARS['outputs_path'] + VARS['flnLogMagSpectrogram']).squeeze()
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
fig, ax = plt.subplots(1, 3, figsize=(30,3))
ax[0].imshow(ccens_content)
ax[0].set_title('content')
ax[1].imshow(ccens_style)
ax[1].set_title('style')
ax[2].imshow(ccens_Y)
ax[2].set_title('mixture')
plt.show()

#=======================================================================================================================
# DEPRECATED BUT WORKS -> uncomment to see unmeaningful metrics but pretty plots :)
im_interested_in_the_plots = False
if im_interested_in_the_plots:
    # still on content - rythm using beat spectrum (Foote 2001)
    # http://www.rotorbrain.com/foote/papers/icme2001/icmehtml.htm#pgfId-1000011644
    # 1- compute self similarity matrices (we'll do that on the log|STFT|)
    assert Y_logstft.shape[0] == content_logstft.shape[0] == style_logstft.shape[0]
    L = Y_logstft.shape[0]
    norm_Y = np.sqrt(np.sum(Y_logstft ** 2, axis=1, keepdims=True))  # column vector
    norm_content = np.sqrt(np.sum(content_logstft ** 2, axis=1, keepdims=True))  # column vector
    norm_style = np.sqrt(np.sum(style_logstft ** 2, axis=1, keepdims=True))  # column vector
    ssm_Y = Y_logstft[:,:Tmin] @ Y_logstft[:,:Tmin].T / (norm_Y @ norm_Y.T)
    ssm_content = content_logstft[:,:Tmin] @ content_logstft[:,:Tmin].T / (norm_content @ norm_content.T)
    ssm_style = style_logstft[:,:Tmin] @ style_logstft[:,:Tmin].T / (norm_style @ norm_style.T)
    # visualize self-similarity matrices (SSM), mainly for debugging
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(ssm_content)
    ax[0].set_title('content')
    ax[1].imshow(ssm_style)
    ax[1].set_title('style')
    ax[2].imshow(ssm_Y)
    ax[2].set_title('mixture')
    plt.show()
    # compute beat spectrum
    bs_Y = []
    bs_content = []
    bs_style = []
    for lag in range(L):
        # compute SUM_i(SSM[i,i+lag]) and add to the list
        bs_Y.append(np.trace(ssm_Y, offset=lag))
        bs_content.append(np.trace(ssm_content, offset=lag))
        bs_style.append(np.trace(ssm_style, offset=lag))
    # visualize beat spectra
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(bs_content)
    ax[0].set_title('content')
    ax[1].plot(bs_style)
    ax[1].set_title('style')
    ax[2].plot(bs_Y)
    ax[2].set_title('mixture')
    plt.show()
    # compare the beat spectra as a single feature
    bs_Y = np.array(bs_Y)
    bs_content = np.array(bs_content)
    bs_style = np.array(bs_style)
    norm_Y = np.sqrt(np.sum(bs_Y * bs_Y))
    norm_content = np.sqrt(np.sum(bs_content * bs_content))
    norm_style = np.sqrt(np.sum(bs_style * bs_style))
    sim_Y_content = np.dot(bs_Y, bs_content) / (norm_Y * norm_content)
    sim_Y_style = np.dot(bs_Y, bs_style) / (norm_Y * norm_style)
    sim_content_style = np.dot(bs_content, bs_style) / (norm_content * norm_style)  # expected to be small
    print('COSINE SIMILARITIES BETWEEN BEAT SPECTRA:')
    print('             content   style     mixture')
    print('content {:10.4f}{:10.4f}{:10.4f}'.format(1, sim_content_style, sim_Y_content))
    print('style   {:10.4f}{:10.4f}{:10.4f}'.format(sim_content_style, 1, sim_Y_style))
    print('mixture {:10.4f}{:10.4f}{:10.4f}'.format(sim_Y_content, sim_Y_style, 1))

    # evaluate style with MFCC
    mfcc_Y = librosa.feature.mfcc(Y_audio[:Tmin], sr=args['sr'], hop_length=args['hoplen'], n_mfcc=13)
    mfcc_content = librosa.feature.mfcc(content[:Tmin], sr=args['sr'], hop_length=args['hoplen'], n_mfcc=13)
    mfcc_style = librosa.feature.mfcc(style[:Tmin], sr=args['sr'], hop_length=args['hoplen'], n_mfcc=13)
    # cosine similarity between chroma cens
    norm_Y = np.sqrt(np.sum(mfcc_Y * mfcc_Y))
    norm_content = np.sqrt(np.sum(mfcc_content * mfcc_content))
    norm_style = np.sqrt(np.sum(mfcc_style * mfcc_style))
    sim_Y_content = np.dot(mfcc_Y.flatten(), mfcc_content.flatten()) / (norm_Y * norm_content)
    sim_Y_style = np.dot(mfcc_Y.flatten(), mfcc_style.flatten()) / (norm_Y * norm_style)
    sim_content_style = np.dot(mfcc_content.flatten(), mfcc_style.flatten()) / (norm_content * norm_style)  # expected to be small
    print('COSINE SIMILARITIES BETWEEN MFCCs:')
    print('             content   style     mixture')
    print('content {:10.4f}{:10.4f}{:10.4f}'.format(1, sim_content_style, sim_Y_content))
    print('style   {:10.4f}{:10.4f}{:10.4f}'.format(sim_content_style, 1, sim_Y_style))
    print('mixture {:10.4f}{:10.4f}{:10.4f}'.format(sim_Y_content, sim_Y_style, 1))
    # visualize MFCCS
    fig, ax = plt.subplots(1, 3, figsize=(30,3))
    ax[0].imshow(mfcc_content)
    ax[0].set_title('content')
    ax[1].imshow(mfcc_style)
    ax[1].set_title('style')
    ax[2].imshow(mfcc_Y)
    ax[2].set_title('mixture')
    plt.show()



