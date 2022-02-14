"""This file is a modified version of https://github.com/sigsep/open-unmix-pytorch/blob/master/openunmix/data.py
It contains the dataset classes used for training and testing"""

from .utils import load_audio, load_info
from pathlib import Path
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import librosa as lb

import glob
import argparse
import random
import musdb
import torch
import tqdm
import os
import pickle
import csv
import json


class MixSNR(object):
    """
    ..........
    """

    def __init__(self):
        pass

    def __call__(self, target_snr, vocals, accompaniment, vocals_start=0, vocals_length=None, sil=False):
        if vocals_length is None:
            vocals_length = vocals.size()[1]

        vocals_energy = torch.sum(torch.mean(vocals, dim=0) ** 2)
        # compute accompaniment energy only on part where vocals are present
        acc_energy = torch.sum(torch.mean(accompaniment[:, vocals_start: vocals_start + vocals_length], dim=0) ** 2)
            
        # if acc_energy > 0.1 and vocals_energy > 0.1:
        if not sil:
            snr_current = 10 * torch.log10(vocals_energy / acc_energy)
            snr_difference = target_snr - snr_current
            scaling = (10 ** (snr_difference / 10))
            vocals_scaled = vocals * torch.sqrt(scaling)
            mix = vocals_scaled + accompaniment
            mix_max = abs(mix).max()
            mix = mix / mix_max
            vocals_scaled = vocals_scaled / mix_max
            accompaniment = accompaniment / mix_max
            return mix, vocals_scaled, accompaniment
        else:
            mix = vocals + accompaniment
            mix_max = abs(mix).max()
            mix = mix / mix_max
            vocals = vocals / mix_max
            accompaniment = accompaniment / mix_max
            return mix, vocals, accompaniment
class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g
def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio

class TIMITMusicTrain(torch.utils.data.Dataset):

    def __init__(self,
                 text_units,
                 fixed_length=False,
                 space_token_only=False,
                 mono=False):

        super(TIMITMusicTrain).__init__()

        # set to true if all files should have fixed length of 8.5 s (=> much silence in true vocals)
        self.fixed_length = fixed_length
        self.text_units = text_units
        self.space_token_only = space_token_only
        self.mono = mono
        self.sample_rate = 16000  # TIMIT is only available at 16 kHz
        with open('location_dict.json') as f:
            self.addr_dict = json.load(f)

        self.data_set_root = os.path.join(self.addr_dict["dataset_root"], 'lmbmm_vocal_sep_data/TIMIT/train/')

        if text_units == 'cmu_phonemes':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'cmu_phoneme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 44
        if text_units == 'ones':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'cmu_phoneme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 44
        if text_units == 'visemes':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'viseme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 24
        if text_units == "landmarks":
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'viseme_sequences_idx_open_unmix/train')
        if text_units == None:
            self.path_to_text_sequences = None

        # music related
        # os.path.join(self.addr_dict["dataset_root"], "Music_instrumentals/train/torch_snippets")
        path_to_music = os.path.join(self.addr_dict["dataset_root"], "lmbmm_vocal_sep_data/INSTRUMENT/data/")
        self.list_of_music_files = sorted([f for f in glob.glob(path_to_music + "/*.pt", recursive=True)])
        self.mix_with_snr = MixSNR()

    def __len__(self):
        return 4560  # number of TIMIT utterances assigned to training set

    def __getitem__(self, idx):
        # get speech file os.path.join(self.addr_dict["dataset_root"], 'TIMIT/TIMIT_torch/train/{}.pt'.format(idx))


        speech = torch.load(os.path.join(self.data_set_root, '{}.pt'.format(idx)))
        if self.mono:
            speech = speech.unsqueeze(0)
            speech = speech.tile((2,1))
        # randomly choose a music file from list and load it
        music_idx = torch.randint(low=0, high=len(self.list_of_music_files), size=(1,))
        music_file = self.list_of_music_files[music_idx]
        music = torch.load(os.path.join(self.data_set_root, music_file))

        if self.fixed_length:
            # pad the speech signal to same length as music
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            padding_at_start = int(torch.randint(0, music_len - speech_len, size=(1,)))
            padding_at_end = music_len - padding_at_start - speech_len
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)

        else:
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            max_pad = min((music_len - speech_len) // 2, self.sample_rate)
            padding_at_start = int(torch.randint(0, max_pad, size=(1,)))
            padding_at_end = int(torch.randint(0, max_pad, size=(1,)))
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)
            music = music[:, 0:speech_padded.shape[1]]
        if not self.path_to_text_sequences is None:
            side_info = torch.load(os.path.join(self.path_to_text_sequences, '{}.pt'.format(idx)))
            if self.space_token_only:
                # put space token instead of silence token at start and end of text sequence
                # this option should not be used for pre-training on speech,
                # otherwise the alinment will not be learned
                # when training on MUSDB and added spech examples, this option can be selected
                # (training on MUSDB without pre-training and using the silence token
                # does not enable learning the alignment)
                side_info[0] = 3
                side_info[-1] = 3
        else:
            # this will stay like this for now until I implement the data
            side_info = torch.ones_like(music)

        if self.text_units == 'ones':
            side_info = torch.ones_like(side_info)

        target_snr = torch.rand(size=(1,)) * (-8)
        mix, speech, music = self.mix_with_snr(target_snr, torch.from_numpy(speech_padded).type(torch.float32),
                                               music, padding_at_start, speech_len)

        return mix, speech, side_info
class TIMITMusicTest(torch.utils.data.Dataset):

    def __init__(self,
                 text_units,
                 fixed_length=False,
                 space_token_only=False,
                 mono=False,
                 size=1344):

        super(TIMITMusicTest).__init__()

        # set to true if all files should have fixed length of 8.5 s (=> much silence in true vocals)
        self.fixed_length = fixed_length
        self.text_units = text_units
        self.space_token_only = space_token_only
        self.mono = mono
        self.data_set_size = size
        self.sample_rate = 16000  # TIMIT is only available at 16 kHz
        with open('location_dict.json') as f:
            self.addr_dict = json.load(f)

        self.data_set_root = os.path.join(self.addr_dict["dataset_root"], 'lmbmm_vocal_sep_data/TIMIT/test/')

        if text_units == 'cmu_phonemes':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'cmu_phoneme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 44
        if text_units == 'ones':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'cmu_phoneme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 44
        if text_units == 'visemes':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'viseme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 24
        if text_units == "landmarks":
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'viseme_sequences_idx_open_unmix/train')
        if text_units == None:
            self.path_to_text_sequences = None

        # music related
        # os.path.join(self.addr_dict["dataset_root"], "Music_instrumentals/train/torch_snippets")
        path_to_music = os.path.join(self.addr_dict["dataset_root"], "lmbmm_vocal_sep_data/INSTRUMENT/data/")
        self.list_of_music_files = sorted([f for f in glob.glob(path_to_music + "/*.pt", recursive=True)])
        self.mix_with_snr = MixSNR()

    def __len__(self):
        return min(1344, self.data_set_size)  # number of TIMIT utterances assigned to training set

    def __getitem__(self, idx):
        # get speech file os.path.join(self.addr_dict["dataset_root"], 'TIMIT/TIMIT_torch/train/{}.pt'.format(idx))


        speech = torch.load(os.path.join(self.data_set_root, '{}.pt'.format(idx)))
        if self.mono:
            speech = speech.unsqueeze(0)
            speech = speech.tile((2,1))
        # randomly choose a music file from list and load it
        music_idx = torch.randint(low=0, high=len(self.list_of_music_files), size=(1,))
        music_file = self.list_of_music_files[music_idx]
        music = torch.load(os.path.join(self.data_set_root, music_file))

        if self.fixed_length:
            # pad the speech signal to same length as music
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            padding_at_start = int(torch.randint(0, music_len - speech_len, size=(1,)))
            padding_at_end = music_len - padding_at_start - speech_len
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)

        else:
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            max_pad = min((music_len - speech_len) // 2, self.sample_rate)
            padding_at_start = int(torch.randint(0, max_pad, size=(1,)))
            padding_at_end = int(torch.randint(0, max_pad, size=(1,)))
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)
            music = music[:, 0:speech_padded.shape[1]]
        if not self.path_to_text_sequences is None:
            side_info = torch.load(os.path.join(self.path_to_text_sequences, '{}.pt'.format(idx)))
            if self.space_token_only:
                # put space token instead of silence token at start and end of text sequence
                # this option should not be used for pre-training on speech,
                # otherwise the alinment will not be learned
                # when training on MUSDB and added spech examples, this option can be selected
                # (training on MUSDB without pre-training and using the silence token
                # does not enable learning the alignment)
                side_info[0] = 3
                side_info[-1] = 3
        else:
            # this will stay like this for now until I implement the data
            side_info = torch.ones_like(music)

        if self.text_units == 'ones':
            side_info = torch.ones_like(side_info)

        target_snr = torch.rand(size=(1,)) * (-8)
        mix, speech, music = self.mix_with_snr(target_snr, torch.from_numpy(speech_padded).type(torch.float32),
                                               music, padding_at_start, speech_len)

        return mix, speech, side_info

class NUSMusicTrain(torch.utils.data.Dataset):

    def __init__(self,
                 text_units,
                 fixed_length=False,
                 space_token_only=False,
                 mono=False):

        super(NUSMusicTrain).__init__()

        # set to true if all files should have fixed length of 8.5 s (=> much silence in true vocals)
        self.fixed_length = fixed_length
        self.text_units = text_units
        self.space_token_only = space_token_only
        self.mono = mono
        self.sample_rate = 16000  # TIMIT is only available at 16 kHz
        with open('location_dict.json') as f:
            self.addr_dict = json.load(f)

        self.data_set_root = os.path.join(self.addr_dict["dataset_root"], 'lmbmm_vocal_sep_data/TIMIT/train/')

        if text_units == 'cmu_phonemes':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'cmu_phoneme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 44
        if text_units == 'ones':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'cmu_phoneme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 44
        if text_units == 'visemes':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'viseme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 24
        if text_units == "landmarks":
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'viseme_sequences_idx_open_unmix/train')
        if text_units == None:
            self.path_to_text_sequences = None

        # music related
        # os.path.join(self.addr_dict["dataset_root"], "Music_instrumentals/train/torch_snippets")
        path_to_music = os.path.join(self.addr_dict["dataset_root"], "lmbmm_vocal_sep_data/INSTRUMENT/data/")
        self.list_of_music_files = sorted([f for f in glob.glob(path_to_music + "/*.pt", recursive=True)])
        self.mix_with_snr = MixSNR()

    def __len__(self):
        return 738  # number of TIMIT utterances assigned to training set

    def __getitem__(self, idx):
        # get speech file os.path.join(self.addr_dict["dataset_root"], 'TIMIT/TIMIT_torch/train/{}.pt'.format(idx))


        speech = torch.load(os.path.join(self.data_set_root, '{}.pt'.format(idx)))
        if self.mono:
            speech = speech.unsqueeze(0)
            speech = speech.tile((2,1))
        # randomly choose a music file from list and load it
        music_idx = torch.randint(low=0, high=len(self.list_of_music_files), size=(1,))
        music_file = self.list_of_music_files[music_idx]
        music = torch.load(os.path.join(self.data_set_root, music_file))

        if self.fixed_length:
            # pad the speech signal to same length as music
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            padding_at_start = int(torch.randint(0, music_len - speech_len, size=(1,)))
            padding_at_end = music_len - padding_at_start - speech_len
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)

        else:
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            max_pad = min((music_len - speech_len) // 2, self.sample_rate)
            padding_at_start = int(torch.randint(0, max_pad, size=(1,)))
            padding_at_end = int(torch.randint(0, max_pad, size=(1,)))
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)
            music = music[:, 0:speech_padded.shape[1]]
        if not self.path_to_text_sequences is None:
            side_info = torch.load(os.path.join(self.path_to_text_sequences, '{}.pt'.format(idx)))
            if self.space_token_only:
                # put space token instead of silence token at start and end of text sequence
                # this option should not be used for pre-training on speech,
                # otherwise the alinment will not be learned
                # when training on MUSDB and added spech examples, this option can be selected
                # (training on MUSDB without pre-training and using the silence token
                # does not enable learning the alignment)
                side_info[0] = 3
                side_info[-1] = 3
        else:
            # this will stay like this for now until I implement the data
            side_info = torch.ones_like(music)

        if self.text_units == 'ones':
            side_info = torch.ones_like(side_info)

        target_snr = torch.rand(size=(1,)) * (-8)
        mix, speech, music = self.mix_with_snr(target_snr, torch.from_numpy(speech_padded).type(torch.float32),
                                               music, padding_at_start, speech_len)

        return mix, speech, side_info
class NUSMusicTest(torch.utils.data.Dataset):

    def __init__(self,
                 text_units,
                 fixed_length=False,
                 space_token_only=False,
                 mono=False,
                 size=1344):

        super(NUSMusicTest).__init__()

        # set to true if all files should have fixed length of 8.5 s (=> much silence in true vocals)
        self.fixed_length = fixed_length
        self.text_units = text_units
        self.space_token_only = space_token_only
        self.mono = mono
        self.data_set_size = size
        self.sample_rate = 16000  # TIMIT is only available at 16 kHz
        with open('location_dict.json') as f:
            self.addr_dict = json.load(f)

        self.data_set_root = os.path.join(self.addr_dict["dataset_root"], 'lmbmm_vocal_sep_data/TIMIT/test/')

        if text_units == 'cmu_phonemes':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'cmu_phoneme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 44
        if text_units == 'ones':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'cmu_phoneme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 44
        if text_units == 'visemes':
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'viseme_sequences_idx_open_unmix/train')
            self.vocabulary_size = 24
        if text_units == "landmarks":
            self.path_to_text_sequences = os.path.join(self.addr_dict["dataset_root"],
                                                       'viseme_sequences_idx_open_unmix/train')
        if text_units == None:
            self.path_to_text_sequences = None

        # music related
        # os.path.join(self.addr_dict["dataset_root"], "Music_instrumentals/train/torch_snippets")
        path_to_music = os.path.join(self.addr_dict["dataset_root"], "lmbmm_vocal_sep_data/INSTRUMENT/data/")
        self.list_of_music_files = sorted([f for f in glob.glob(path_to_music + "/*.pt", recursive=True)])
        self.mix_with_snr = MixSNR()

    def __len__(self):
        return min(248, self.data_set_size)  # number of TIMIT utterances assigned to training set

    def __getitem__(self, idx):
        # get speech file os.path.join(self.addr_dict["dataset_root"], 'TIMIT/TIMIT_torch/train/{}.pt'.format(idx))


        speech = torch.load(os.path.join(self.data_set_root, '{}.pt'.format(idx)))
        if self.mono:
            speech = speech.unsqueeze(0)
            speech = speech.tile((2,1))
        # randomly choose a music file from list and load it
        music_idx = torch.randint(low=0, high=len(self.list_of_music_files), size=(1,))
        music_file = self.list_of_music_files[music_idx]
        music = torch.load(os.path.join(self.data_set_root, music_file))

        if self.fixed_length:
            # pad the speech signal to same length as music
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            padding_at_start = int(torch.randint(0, music_len - speech_len, size=(1,)))
            padding_at_end = music_len - padding_at_start - speech_len
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)

        else:
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            max_pad = min((music_len - speech_len) // 2, self.sample_rate)
            padding_at_start = int(torch.randint(0, max_pad, size=(1,)))
            padding_at_end = int(torch.randint(0, max_pad, size=(1,)))
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)
            music = music[:, 0:speech_padded.shape[1]]
        if not self.path_to_text_sequences is None:
            side_info = torch.load(os.path.join(self.path_to_text_sequences, '{}.pt'.format(idx)))
            if self.space_token_only:
                # put space token instead of silence token at start and end of text sequence
                # this option should not be used for pre-training on speech,
                # otherwise the alinment will not be learned
                # when training on MUSDB and added spech examples, this option can be selected
                # (training on MUSDB without pre-training and using the silence token
                # does not enable learning the alignment)
                side_info[0] = 3
                side_info[-1] = 3
        else:
            # this will stay like this for now until I implement the data
            side_info = torch.ones_like(music)

        if self.text_units == 'ones':
            side_info = torch.ones_like(side_info)

        target_snr = torch.rand(size=(1,)) * (-8)
        mix, speech, music = self.mix_with_snr(target_snr, torch.from_numpy(speech_padded).type(torch.float32),
                                               music, padding_at_start, speech_len)

        return mix, speech, side_info