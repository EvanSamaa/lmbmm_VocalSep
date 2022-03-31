import torch
import torch.nn as nn
import numpy as np
import python_speech_features as psf
import librosa
from matplotlib import pyplot as plt

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from collections import OrderedDict
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class vowel_mod_detector():
    def __init__(self, model_dir = "./trained_models/over_trained_model.pt"):
        if torch.cuda.is_available():
            dev = "cuda:1"
        else:
            dev = "cpu"
        self.model = LSTM_vowel_recognizer_no_BN()
        self.model.load_state_dict(torch.load(model_dir, map_location=torch.device(dev))["model_state_dict"])
    def __call__(self, sound_arr, *args, **kwargs):
        # audio should be the
        sound_arr = (sound_arr - sound_arr.mean()) / sound_arr.std()
        sr = 44100
        appended = False
        if sound_arr.shape[0] < sr * 0.2:
            sound_arr = np.concatenate([np.zeros((int(sr * 0.2), )), sound_arr], axis=0)
            appended = True
        winstep = 441
        mfcc_feat = psf.mfcc(sound_arr, samplerate=sr, winlen=0.02, nfft=2 * winstep, numcep=13)
        logfbank_feat = psf.logfbank(sound_arr, samplerate=sr, winlen=0.02, nfft=2 * winstep, nfilt=26)
        ssc_feat = psf.ssc(sound_arr, samplerate=sr, winlen=0.02, nfft=2 * winstep, nfilt=26)
        full_feat = np.concatenate([mfcc_feat, logfbank_feat, ssc_feat], axis=1)
        input_vec = torch.tensor(np.expand_dims(full_feat, axis=0), dtype=torch.float32)
        out_vec = self.model.test_forward(input_vec)[0]
        if appended:
            out = out_vec.detach().numpy()[20:]
        else:
            out = out_vec.detach().numpy()
        coarse_out = out[:, 0:4]
        coarse_out[:, 1] = out[:, 1] + out[:, 2]
        coarse_out[:, 2] = out[:, 3] + out[:, 4]
        coarse_out[:, 3] = out[:, 5]
        return out, coarse_out
class LSTM_vowel_recognizer_no_BN(nn.Module):
    def __init__(self, hidden_dim=256, win_length=12, num_lstm_layer=3):
        super(LSTM_vowel_recognizer_no_BN, self).__init__()
        # here I'm going to assume the input is gonna be shaped as
        self.hidden_dim = hidden_dim
        self.win_length = win_length
        self.num_lstm_layer = num_lstm_layer
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(65 * (self.win_length * 2 + 1), self.hidden_dim, num_lstm_layer, batch_first=True)
        self.output_mat1 = nn.Linear(self.hidden_dim, 6)
        self.output_mat2 = nn.Linear(64, 64)
        self.output_mat3 = nn.Linear(64, 6)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
    def concate_frames(self, input_audio):
        padding = torch.zeros((input_audio.shape[0], self.win_length, input_audio.shape[2]))
        padded_input_audio = torch.cat([padding, input_audio, padding], dim=1)
        window_audio = []
        for i in range(0, input_audio.shape[1]):
            window_count = i + 12
            current_window = padded_input_audio[:, window_count-12:window_count+13]
            s = current_window.shape
            current_window = current_window.view((s[0], s[1] * s[2]))
            current_window = torch.unsqueeze(current_window, 1)
            window_audio.append(current_window)
        rtv = torch.cat(window_audio, dim=1)
        return rtv
    def forward(self, input_audio):
        mod_audio = self.concate_frames(input_audio)
        # here I'm assuming that the input_audio is of shape
        hidden_state = [torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)), torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim))]
        out, hidden_state = self.lstm(mod_audio, hidden_state)
        # bn
        # x = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(out)
        x = self.output_mat1(x)
        return x

    def test_forward(self, input_audio):
        x = self.forward(input_audio)
        return self.softmax(x)