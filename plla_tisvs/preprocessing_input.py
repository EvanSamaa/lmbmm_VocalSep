import parselmouth
import pickle
import os
import numpy as np
from scipy.signal import decimate
import torch
import json
from . import testx
from .estimate_alignment import optimal_alignment_path
from .estimate_alignment import compute_phoneme_onsets
import librosa
from scipy.io import wavfile

class Custom_data_set():
    def __init__(self, dict_path, phoneme_dict_path):
        # build dictionary to transformed between phoneme to the indexes used in the paper
        cmu_vocabulary = ['#', '$', '%', '>', '-', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
                      'EY', 'F', 'G',
                      'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH',
                      'UH',
                      'UW', 'V', 'W', 'Y', 'Z', 'ZH']
        self.CMU2VISEME = {"AA": "A", "AO": "A", "AY": "A", "AW": "A", "AE": "E",
                      "EY": "E", "UH": "A", "UW": "U", "IH": "I", "IY": "I", "EH": "E", "HH": "E", "UH": "U", "AH": "E",
                      "ER": "E", "OW": "O", "OY": "O", "R": "R", "D": "LNTD", "T": "LNTD", "L": "LNTD", "N": "LNTD",
                      "NG": "LNTD",
                      "F": "FV", "V": "FV", "B": "BP", "M": "M", "P": "BP", "CH": "ShChZh", "SH": "ShChZh",
                      "ZH": "ShChZh",
                      "S": "SZ", "Z": "SZ", "DH": "Th", "TH": "Th", "G": "GK", "K": "GK", "Y": "Y", "JH": "J", "W": "W",
                      '#': '#',
                      '$': '$', '%': '%', '>': '>', '-': '-'}
        visemes_vocabulary = ['#', '$', '%', '>', '-', 'M', 'BP', "Y", "J", "R", "FV", "LNTD", "M", "BP", "W", "Th",
                              "GK",
                              "ShChZh", "SZ", "A", "E", "I", "O", "U"]
        self.cmu_phoneme2idx = {}
        self.cmu_idx2phoneme = {}
        self.viseme2idx = {}
        self.idx2viseme = {}
        for idx, phoneme in enumerate(cmu_vocabulary):
            self.cmu_phoneme2idx[phoneme] = idx
            self.cmu_idx2phoneme[idx] = phoneme

        for idx, viseme in enumerate(visemes_vocabulary):
            self.viseme2idx[viseme] = idx
            self.idx2viseme[idx] = viseme
        with open(os.path.join(dict_path, phoneme_dict_path), "rb") as file:
            self.word2phoneme_dict = pickle.load(file)
        with open(os.path.join(dict_path, "cmu_symbols2phones.pickle"), "rb") as file:
            self.symbol2phoneme_dict = pickle.load(file)

    def parse(self, audio_path, transcirpt_path, vocab="cmu_phonemes"):
        phoneme_list = []

        # load the transcript file
        with open(transcirpt_path, "r") as file:
            transcript = file.read()

        # replace useless symbols
        transcript = transcript.replace("\n", " ")
        transcript = transcript.replace(",", "")
        transcript = transcript.replace("?", "")
        transcript = transcript.replace(".", "")
        transcript = transcript.replace("(", "")
        transcript = transcript.replace(")", "")
        transcript = transcript.replace("-", " ")

        phoneme_list_full = []
        word_list = []
        for word in transcript.split():
            phonemes = self.word2phoneme_dict[word.lower()]
            word_list.append(word.lower())
            for phoneme in phonemes.split():
                phoneme_list.append(self.symbol2phoneme_dict[phoneme])
                phoneme_list_full.append(">")
                phoneme_list_full.append(phoneme)
                phoneme_list.append(">")
            phoneme_list_full.append("EOW")
        phoneme_list = phoneme_list[:-1]
        if vocab == "cmu_phonemes":
            phoneme_idx = np.array([self.cmu_phoneme2idx[p] for p in phoneme_list])
        else:
            phoneme_idx = np.array([self.viseme2idx[self.CMU2VISEME[p]] for p in phoneme_list])
        phoneme_idx = np.pad(phoneme_idx, (1, 1), mode='constant', constant_values=1)
        phoneme_idx_torch = torch.from_numpy(phoneme_idx)

        # load the audio file into a readable format for the model
        if audio_path[-3:] != "wav":
            print("The file needs to be a wav file for this to work. Try again!")
            raise TypeError
        # resample if neede

        sound_object = parselmouth.Sound(audio_path)
        data = sound_object.values
        sound = []
        samplerate = sound_object.sampling_frequency
        for k in range(0, data.shape[0]):
            sound.append(librosa.resample(data[k, :], samplerate, 16000))
        sound = np.array(sound)
        if vocab == "visemes":
            sound = (sound - sound.mean()) / sound.std()
        else:
            sound = (sound - sound.mean()) / sound.std() * 0.06

        sound_torch = torch.from_numpy(sound.copy()).type(torch.float32)
        sound_torch_out = sound_torch.unsqueeze(dim=0)
        phoneme_idx_out = phoneme_idx_torch.unsqueeze(dim=0)
        return sound_torch_out, phoneme_idx_out, phoneme_list_full, word_list
    def get_phonemes(self, idx_list):
        # input should be an 1D array of indexes, it will be turned into a list of phonemes
        out = []
        for i in range(0, idx_list.size()[0]):
            out.append(self.cmu_idx2phoneme[int(idx_list[i].item())])
        return out

    def get_visemes(self, idx_list):
        # input should be an 1D array of indexes, it will be turned into a list of phonemes
        out = []
        for i in range(0, idx_list.size()[0]):
            out.append(self.idx2viseme[int(idx_list[i].item())])
        return out


if __name__ == "__main__":


    dict_path = "./dicts"
    phoneme_dict_path = "cmu_word2cmu_phoneme_extra.pickle"
    audio_paths = ["/Volumes/EVAN_DISK/ten_videos/I_dont_love_you/I_dont_love_you_short/audio.wav"]
    transcript_paths = ["/Volumes/EVAN_DISK/ten_videos/I_dont_love_you/I_dont_love_you_short/audio.txt"]

    data_parser = Custom_data_set(dict_path, phoneme_dict_path)
    audio, phoneme_idx, phoneme_list_full, word_list = data_parser.parse(audio_paths[0], transcript_paths[0])
    print(phoneme_idx.size)
    # load model
    model_path = 'trained_models/{}'.format("JOINT3")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print("Device:", device)
    target = 'vocals'

    # load model
    model_to_test = testx.load_model(target, model_path, device)
    model_to_test.return_alphas = True
    model_to_test.eval()

    # load model config
    with open(os.path.join(model_path, target + '.json'), 'r') as stream:
        config = json.load(stream)
        samplerate = config['args']['samplerate']
        text_units = config['args']['text_units']
        nfft = config['args']['nfft']
        nhop = config['args']['nhop']

    with torch.no_grad():
        vocals_estimate, alphas, scores = model_to_test((audio, phoneme_idx))

    optimal_path_scores = optimal_alignment_path(scores, mode='max_numpy', init=200)

    phoneme_onsets = compute_phoneme_onsets(optimal_path_scores, hop_length=nhop, sampling_rate=samplerate)
    print(len(phoneme_onsets))
    print()



