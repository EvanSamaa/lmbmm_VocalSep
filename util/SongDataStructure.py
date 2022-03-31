import librosa


import os
import math
import time
# import winsound
import textgrids
import parselmouth
from scipy.signal import savgol_filter, correlate, convolve
from scipy.signal.windows import gaussian
from scipy.interpolate import interp1d
import numpy as np

import torch
from plla_tisvs.preprocessing_input import Custom_data_set
import plla_tisvs.testx as testx
from plla_tisvs.estimate_alignment import optimal_alignment_path, compute_phoneme_onsets
import json
import re
from util.pitch_interval_estimation import *


NOTES_NAME = ["A", "A#", "B", "C", "C#", "D",
              "D#", "E", "F", "F#", "G", "G#"]
NOTES_DICT = {"A": 0, "A#": 1, "B": 2, "C": 3, "C#": 4, "D": 5,
              "D#": 6, "E": 7, "F": 8, "F#": 9, "G": 10, "G#": 11}
LOWEST_NOTE = 27.50
RESTING_FREQUENCY = 60

SOPRANO = ["C4", "A5"]
MEZZO_SOPRANO = ["A3", "F5"]
ALTO = ["G3", "E5"]
TENOR = ["C3", "A4"]
BARITONE = ["A2", "F4"]
BASS = ["F2", "E4"]
VOWELS = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', "ER", "N"])

VOCAL_RANGES = [BASS, BARITONE, TENOR, ALTO, MEZZO_SOPRANO, SOPRANO]
VOCAL_RANGES_VAL = [[87.30705785825097, 329.62755691287],
                    [55.0, 349.2282314330039],
                    [130.8127826502993, 220.0],
                    [195.99771799087466, 329.62755691287],
                    [110.0, 698.4564628660079],
                    [261.6255653005986, 440.0]]
VOCAL_RANGES_NAME = ['BASS', 'BARITONE', 'TENOR', 'ALTO', 'MEZZO_SOPRANO', 'SOPRANO']

class Minimal_song_data_structure():
    def __init__(self, audio_path_file, transcript_path, txt_grid_path = "", pitch_ceiling = 1400, alignment_type = "cmu_phonemes", use_torch=False, sr=16000):
        # the audio file should be 44.1kHz for accurate pitch prediction result.
        # the audio file could be a mp3 file
        self.transcript_path = transcript_path
        self.audio_path_file = audio_path_file
        self.alignment_type = alignment_type
        # set some hyperparameters
        self.pitch_ceiling = pitch_ceiling
        self.dt = 0.01
        self.silence_threshold = 0.007
        # obtain sound related data using Praat
        if use_torch:
            # here we assume that if we take in torch data, the sampling frequency will be 16000, but this could be
            # be changed if needed
            temp_snd_arr = torch.load(audio_path_file).detach().numpy()
            # print(temp_snd_arr.shape[0]/16000.0)
            temp_snd_arr = librosa.resample(temp_snd_arr, orig_sr=16000, target_sr=44100)
            self.snd = parselmouth.Sound(temp_snd_arr, sampling_frequency=44100, start_time=0)
        else:
            self.snd = parselmouth.Sound(audio_path_file)
        self.sound_arr = self.snd.as_array()[0]
        # self.sound_arr = librosa.load(audio_path_file, sr = 44100)[0]
        self.sound_arr_interp = interp1d(self.snd.xs(), self.sound_arr)

        # librosa.load(audio_path_file, sr=44100)[0]

        self.pitch = self.snd.to_pitch(time_step = self.dt, pitch_ceiling = self.pitch_ceiling)
        self.pitch_arr = self.pitch.selected_array["frequency"]
        if np.square(self.pitch_arr).sum() != 0:
            self.pitch_arr[self.pitch_arr == 0] = np.nan
            mask = np.isnan(self.pitch_arr)
            self.pitch_arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.pitch_arr[~mask])
        # use interpolation to deal with missing value in the pitch prediction
        self.pitch_interp = interp1d(self.pitch.xs(), self.pitch_arr)
        self.intensity = self.snd.to_intensity(time_step=self.dt)
        self.intensity_arr = self.intensity.values.T[:, 0]
        self.intensity_interp = interp1d(self.intensity.xs(), self.intensity_arr)
        self.formant = self.snd.to_formant_burg(time_step=self.dt)
        self.formants_arr = np.zeros((len(self.formant.xs()), 3))
        for i in range(0, self.formants_arr.shape[0]):
            for j in range(1, 4):
                self.formants_arr[i, j - 1] = self.formant.get_value_at_time(j, self.formant.xs()[i])
        self.F1_interp = interp1d(self.formant.xs(), self.formants_arr[:,0])
        self.xs = self.pitch.xs()

        # pre-init variable to store voice qualities such as belting/head voice and etc
        self.voice_quality_intervals = []
        self.voice_quality_lists = []
        self.coarse_voice_quality_intervals = []
        self.coarse_voice_quality_lists = []

        # pre-init variable used for storing vibrato information
        self.vibrato_intervals = []
        self.coarse_vibrato_intervals = []

        # list for storing phoneme and word_alignment info
        self.phoneme_list = []
        self.phoneme_list_full = []
        self.phoneme_intervals = []
        self.word_list = []
        self.word_intervals = []

        # list for storing pitch related details
        self.pitch_slopes = []
        self.pitch_intervals = []

        # if these parameters are non-empty then they are called to add to the thing
        if txt_grid_path != "":
            self.phoneme_list, self.phoneme_intervals, self.word_list, self.word_intervals = self.load_phoneme_textgrid(txt_grid_path)
    def get_f_interval(self, interval):
        try:
            ts = np.arange(max(interval[0], self.pitch.xs()[0]), min(interval[1], self.pitch.xs()[-1]), self.dt)
        except:
            ts = np.arange(max(interval[0], self.pitch_xs[0]), min(interval[1], self.pitch_xs[-1]), self.dt)
        return ts, self.pitch_interp(ts)
    def get_I_interval(self, interval):
        ts = np.arange(interval[0], min(interval[1], self.intensity.xs()[-1]), self.dt)
        return ts, self.intensity_interp(ts)
    def get_F1_interval(self, interval):
        ts = np.arange(interval[0], min(interval[1], self.formant.xs()[-1]), self.dt)
        return ts, self.F1_interp(ts)
    def compute_self_vibrato_intervals(self):
        if len(self.vibrato_intervals) == 0:
            strength = self.pitch.selected_array["strength"]
            frequency = self.pitch.selected_array["frequency"]
            frequency[strength < 0.5] = 0
            frequency[frequency == 0] = np.nan
            frequency_xs = self.xs
            self.vibrato_intervals = self.compute_vibrato_intervals(frequency, frequency_xs, self.dt)

        # obtain variables
        xs = self.xs
        freq = self.pitch.selected_array["frequency"]
        # smooth frequency so it does not have sudden jumps during silence
        freq[freq == 0] = np.nan
        # use interpolation to deal with missing value in the pitch prediction
        f = interp1d(xs, freq)


        sub_intervals = self.__get_subarrays_indexes_from_time_interval(self.phoneme_intervals, xs)
        self.vibrato_intervals = []
        for i in range(0, len(sub_intervals)):
            interval = sub_intervals[i]
            phone = self.phoneme_list[i]
            if phone in VOWELS:
                vib_interval = []

                if xs[interval[1]] - xs[interval[0]] > 0.3:
                    xs_phone = xs[interval[0]:interval[1]]
                    temp_vib_interval = self.compute_vibrato_intervals(f(xs_phone), xs_phone, self.dt)

                    # merge these intervals in necessary
                    j = 0
                    while j <= len(temp_vib_interval) - 1:
                        if j == len(temp_vib_interval) - 1:
                            vib_interval.append(temp_vib_interval[j])
                            j = j + 1
                        elif abs(temp_vib_interval[j][1] - temp_vib_interval[j + 1][0]) <= self.dt * 2.5:
                            vib_interval.append([temp_vib_interval[j][0], temp_vib_interval[j + 1][0]])
                            j = j + 2
                        else:
                            vib_interval.append(temp_vib_interval[j])
                            j = j + 1
                    if j == 0:
                        vib_interval = temp_vib_interval
                self.vibrato_intervals.append(vib_interval)
        return self.vibrato_intervals
    def compute_vibrato_intervals_old(self, frequency, frequency_xs, dt):
        # this function computes vibrato by analyzing the zero-crossing of the input frequency array
        # if it can identify 3 zero crossing that are equally spaced apart, then it recognize those
        # as a vibrato.
        min_zero_crossing_distance = 1.0 / 16  # max vibrato frequency = 7 Hz = 14 zero crossings per second
        self.tolerance = dt * 4  # using the uncertainty of the instrument (pitch measuring device) to bound tolerance

        # compute time derivative
        d_frequency_dt = correlate(frequency, np.array([-1.0, 0, 1.0]), mode="same") / dt / 2

        # obtain zero crossings
        zero_crossing = []
        for i in range(0, d_frequency_dt.shape[0] - 1):
            if (d_frequency_dt[i] < 0 and d_frequency_dt[i + 1] >= 0) or (
                    d_frequency_dt[i] > 0 and d_frequency_dt[i + 1] <= 0):
                zero_crossing.append(i + 1)

        # choose sets of zero crossing and identify vibratos within those
        distance = 0
        in_vibrato = 0
        starting_time = -1
        vibrato_intervals = []
        for i in range(0, len(zero_crossing) - 1):
            current_distance = frequency_xs[zero_crossing[i + 1]] - frequency_xs[zero_crossing[i]]

            if abs(current_distance - distance) <= self.tolerance and current_distance >= min_zero_crossing_distance:
                if in_vibrato == 0:
                    starting_time = zero_crossing[i - 1]
                    distance = (distance + current_distance) / 2  # calculate new average
                    in_vibrato = 1
                elif in_vibrato > 0:
                    distance = (distance * in_vibrato + current_distance) / (in_vibrato + 1)  # calculate new average
                    in_vibrato = in_vibrato + 1
            else:
                if in_vibrato > 0:
                    distance = current_distance
                    if in_vibrato > 2:
                        vibrato_intervals.append([frequency_xs[starting_time], frequency_xs[zero_crossing[i]]])
                    in_vibrato = 0
                else:
                    distance = current_distance
        return vibrato_intervals
    def compute_vibrato_intervals(self, frequency, frequency_xs, dt):
        # this function computes vibrato by analyzing the zero-crossing of the input frequency array
        # if it can identify 3 zero crossing that are equally spaced apart, then it recognize those
        # as a vibrato.
        min_zero_crossing_distance = 1.0 / 14  # max vibrato frequency = 7 Hz = 14 zero crossings per second
        tolerance = dt * 4  # using the uncertainty of the instrument (pitch measuring device) to bound tolerance
        min_height = 15
        # compute time derivative
        d_frequency_dt = correlate(frequency, np.array([-1.0, 0, 1.0]), mode="same") / dt / 2
        f_interp = interp1d(frequency_xs, frequency)
        # obtain zero crossings
        zero_crossing = []
        for i in range(0, d_frequency_dt.shape[0] - 1):
            if (d_frequency_dt[i] < 0 and d_frequency_dt[i + 1] >= 0) or (
                    d_frequency_dt[i] > 0 and d_frequency_dt[i + 1] <= 0):
                zero_crossing.append(i + 1)
        # choose sets of zero crossing and identify vibratos within those
        distance = 0
        in_vibrato = 0
        amplitude_avg = 0
        starting_time = -1
        starting_index = -1
        vibrato_intervals = []
        for i in range(0, len(zero_crossing) - 1):
            current_distance = frequency_xs[zero_crossing[i + 1]] - frequency_xs[zero_crossing[i]]
            current_height = abs(
                f_interp(frequency_xs[zero_crossing[i + 1]]) - f_interp(frequency_xs[zero_crossing[i]]))
            if abs(
                    current_distance - distance) <= tolerance and current_distance - min_zero_crossing_distance >= -dt:
                if i == len(zero_crossing) - 2:
                    if in_vibrato > 0:
                        vib_interval_start = frequency_xs[starting_time]
                        vib_interval_end = frequency_xs[zero_crossing[i]]
                        counter = 1
                        freq_time = frequency_xs[starting_time]
                        # now I count backwards, if the distance includes the previous zero crossing, then
                        # I count that zero_crossing as part of the vibrato
                        for k in range(starting_index - 1, -1, -1):
                            if (frequency_xs[zero_crossing[k]] >= freq_time - counter * distance - dt and
                                    frequency_xs[zero_crossing[k]] <= freq_time - (counter + 1) * distance - dt):
                                vib_interval_start = frequency_xs[zero_crossing[k]]
                                counter = counter + 1
                                in_vibrato = in_vibrato + 1
                            else:
                                break
                        if in_vibrato > 2 and amplitude_avg >= min_height:
                            vibrato_intervals.append(
                                [vib_interval_start, frequency_xs[zero_crossing[min(i + 1, len(zero_crossing) - 1)]]])
                        distance = current_distance
                        in_vibrato = 0
                if in_vibrato == 0:
                    starting_time = zero_crossing[i - 1]
                    starting_index = i - 1
                    distance = (distance + current_distance) / 2  # calculate new average
                    amplitude_avg = (current_height + amplitude_avg) / 2
                    in_vibrato = 1
                elif in_vibrato > 0:
                    distance = (distance * in_vibrato + current_distance) / (in_vibrato + 1)  # calculate new average
                    amplitude_avg = (amplitude_avg * in_vibrato + current_height) / (
                                in_vibrato + 1)  # calculate new average
                    in_vibrato = in_vibrato + 1
            else:
                if in_vibrato > 0:
                    vib_interval_start = frequency_xs[starting_time]
                    vib_interval_end = frequency_xs[zero_crossing[i]]
                    counter = 1
                    freq_time = frequency_xs[starting_time]
                    # now I count backwards, if the distance includes the previous zero crossing, then
                    # I count that zero_crossing as part of the vibrato
                    for k in range(starting_index - 1, -1, -1):

                        if (frequency_xs[zero_crossing[k]] >= freq_time - counter * distance - dt and
                                frequency_xs[zero_crossing[k]] <= freq_time - (counter + 1) * distance - dt):
                            vib_interval_start = frequency_xs[zero_crossing[k]]
                            counter = counter + 1
                            in_vibrato = in_vibrato + 1
                        else:
                            break
                    if in_vibrato > 2 and amplitude_avg >= min_height:
                        vibrato_intervals.append([vib_interval_start, frequency_xs[zero_crossing[i]]])
                    distance = current_distance
                    amplitude_avg = current_height
                    in_vibrato = 0
                else:
                    distance = current_distance
                    amplitude_avg = current_height
        return vibrato_intervals
    def compute_self_pitch_intervals(self):
        sigma = 10
        window_short = 1/np.sqrt(np.pi * 2) / sigma * gaussian(sigma * 2, sigma)
        window = 1/np.sqrt(np.pi * 2) / sigma * gaussian(sigma * 4, sigma)
        if len(self.phoneme_list) <= 0:
            if self.alignment_type == "cmu_phonemes":
                self.compute_self_phoneme_alignment()
            elif self.alignment_type == "visemes":
                self.compute_self_viseme_alignment()
        freq = self.pitch.selected_array["frequency"]
        xs = self.xs
        sub_intervals = self.__get_subarrays_indexes_from_time_interval(self.phoneme_intervals, xs)
        freq[freq == 0] = np.nan
        mask = np.isnan(freq)
        freq[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), freq[~mask])
        # use interpolation to deal with missing value in the pitch prediction
        f = interp1d(xs, freq, kind="linear")

        for i in range(0, len(sub_intervals)):
            interval = sub_intervals[i]
            phone = self.phoneme_list[i]
            pitch_feature_per_vowel = []
            pitch_feature_intervals_per_vowel = []
            if phone in VOWELS:
                interval_length = xs[interval[1]] - xs[interval[0]]
                if interval_length <= 0.4:
                    pitch_feature_per_vowel.append(0)
                    pitch_feature_intervals_per_vowel.append([xs[interval[0]], xs[interval[1]]])
                else:
                    xs_interval = xs[interval[0]:interval[1]]
                    freq_interval = f(xs_interval)
                    padded_freq_interval = np.pad(freq_interval, [sigma * 2, sigma * 2], mode="constant", constant_values=[
                        freq_interval[:sigma * 2].mean(), freq_interval[sigma:].mean()])
                    smoothed_freq = convolve(padded_freq_interval, window, mode="same")[sigma * 2:-sigma * 2]
                    pitch_feature_per_vowel, pitch_feature_intervals_per_vowel = efficient_piece_wise_linear_intervals(xs_interval, smoothed_freq)
                    pitch_feature_intervals_per_vowel = [[xs_interval[val[0]], xs_interval[val[1]]] for val in pitch_feature_intervals_per_vowel]
            self.pitch_intervals.append(pitch_feature_intervals_per_vowel)
            self.pitch_slopes.append(pitch_feature_per_vowel)
    def compute_self_singing_style_intervals(self):
        # if len(self.phoneme_list) <= 0:
        #     self.compute_self_phoneme_alignment()
        self.voice_quality_intervals, self.voice_quality_lists = self.compute_singing_style_intervals(self.snd, self.dt, self.phoneme_list, self.phoneme_intervals)
        self.coarse_voice_quality_lists, self.coarse_voice_quality_intervals = self.__compute_coarse_intervals(self.voice_quality_lists, self.voice_quality_intervals)
    def compute_singing_style_intervals(self, sound: parselmouth.Sound, dt, phone_list, phoneme_intervals):
        formant = sound.to_formant_burg(time_step=self.dt)
        xs = formant.xs()
        formant_arr = np.zeros((len(xs), 3))
        for i in range(0, formant_arr.shape[0]):
            for j in range(1, 4):
                formant_arr[i, j - 1] = formant.get_value_at_time(j, xs[i])
        # here I'm going to smooth the formant with the savgol filter to remove te outliers. This is kind of okay because
        # formant do not have high frequency changes like pitch
        smooth_formant_arr = np.zeros(formant_arr.shape)
        for i in range(0, 3):
            smooth_formant_arr[:, i] = savgol_filter(formant_arr[:, i], 21, 3)
        frequency = self.pitch.selected_array["frequency"]
        # H2_F1 is the distance between the second harmonic and the first formant
        # in belting, Formant is used to raise the second harmonics, while singing
        # in head voice would maintain a low F1 despite the higher pitch.
        H2_F1 = np.abs(frequency[:smooth_formant_arr.shape[0]] * 2 - smooth_formant_arr[:, 0])
        f = interp1d(xs, H2_F1)

        # the intervals are computed based on the formant distance. Each vowel is assumed to be sang in a
        # uniformed singing style.

        voice_quality_intervals = []
        voice_quality_lists = []

        # get location of passagio
        q = np.nanpercentile(frequency, [2, 98])
        passagio = q[0] + 0.5 * (q[1] - q[0])
        for i in range(0, len(phoneme_intervals)):
            # for item in phoneme_alignment["phones"]:
            voice_quality_intervals_i = []
            voice_quality_lists_i = []
            if phone_list[i] in VOWELS:
                # iterate through the type of pitch intervals
                if len(self.pitch_slopes[i]) == 0:
                    vowel_span = np.arange(self.phoneme_intervals[i][0], min(self.phoneme_intervals[i][1], xs[-1]),
                                           dt)
                    H2_F1_i = f(vowel_span)
                    if frequency[:smooth_formant_arr.shape[0]].mean() >= passagio:
                        voice_quality_intervals_i.append(
                            [self.phoneme_intervals[i][0], min(self.phoneme_intervals[i][1], xs[-1])])
                        if H2_F1_i.mean() <= 100:
                            voice_quality_lists_i.append("belt")
                        else:
                            voice_quality_lists_i.append("head")
                    else:
                        voice_quality_intervals_i.append(
                            [self.phoneme_intervals[i][0], min(self.phoneme_intervals[i][1], xs[-1])])
                        voice_quality_lists_i.append("chest")
                for j in range(len(self.pitch_slopes[i])):
                    if self.pitch_slopes[i][j] == 0 or True:
                        vowel_span = np.arange(self.pitch_intervals[i][j][0], min(self.pitch_intervals[i][j][1], xs[-1]), dt)
                        H2_F1_i = f(vowel_span)
                        if frequency[:smooth_formant_arr.shape[0]].mean() >= passagio:
                            voice_quality_intervals_i.append([self.pitch_intervals[i][j][0], min(self.pitch_intervals[i][j][1], xs[-1])])
                            if H2_F1_i.mean() <= 100:
                                voice_quality_lists_i.append("belt")
                            else:
                                voice_quality_lists_i.append("head")
                        else:
                            voice_quality_intervals_i.append([self.pitch_intervals[i][j][0], min(self.pitch_intervals[i][j][1], xs[-1])])
                            voice_quality_lists_i.append("chest")
            voice_quality_lists.append(voice_quality_lists_i)
            voice_quality_intervals.append(voice_quality_intervals_i)
        return voice_quality_intervals, voice_quality_lists
    def compute_word_alignment(self, phoneme_onsets, phoneme_list_full):
        word_durations = []
        pointer_i = 0  # this one is for the phoneme_list_full
        pointer_j = 0  # this one is for phoneme_onsets
        begin = phoneme_onsets[pointer_j]
        begin_ptr = pointer_j
        phone_copy = ['EOW'] + phoneme_list_full
        while pointer_j < phoneme_onsets.shape[0]:
            # if we are at the end of a word right now
            if phone_copy[pointer_i] == "EOW":
                word_durations.append([phoneme_onsets[begin_ptr], phoneme_onsets[min(pointer_j + 1, phoneme_onsets.shape[0] - 1)]])
                if pointer_j + 1 == phoneme_onsets.shape[0]:
                    break
                if phoneme_onsets[min(pointer_j + 1, phoneme_onsets.shape[0] - 1)] != "<":
                    # begin = phoneme_onsets[min(pointer_j + 1, phoneme_onsets.shape[0] - 1)]
                    begin_ptr = min(pointer_j + 1, phoneme_onsets.shape[0] - 1)
                    pointer_i = pointer_i + 2
                    pointer_j = pointer_j + 1
                else:
                    # begin = phoneme_onsets[min(pointer_j + 2, phoneme_onsets.shape[0] - 1)]
                    begin_ptr = min(pointer_j + 2, phoneme_onsets.shape[0] - 1)
                    pointer_i = pointer_i + 3
                    pointer_j = pointer_j + 2
            else:
                pointer_i = pointer_i + 1
                pointer_j = pointer_j + 1
        # make it so that the interval for the first space is included in the intervals
        return [[phoneme_onsets[0], word_durations[1][0]]] + word_durations[1:]
    def compute_self_phoneme_alignment(self):
        dict_path = "./plla_tisvs/dicts"
        model_path = './plla_tisvs/trained_models/{}'.format("JOINT3")
        phoneme_dict_path = "cmu_word2cmu_phoneme_extra.pickle"
        # output_path = "E:/ten_videos/Child_in_time/Child_in_time_2"

        if len(self.phoneme_list) > 0:
            return
        # initilize these variables
        self.phoneme_list = []
        self.phoneme_list_full = []
        self.phoneme_intervals = []
        self.word_list = []
        self.word_intervals = []

        # parse data
        try:
            data_parser = Custom_data_set(dict_path, phoneme_dict_path)
        except:
            dict_path = "." + dict_path
            data_parser = Custom_data_set(dict_path, phoneme_dict_path)
        audio, phoneme_idx, phoneme_list_full, self.word_list = data_parser.parse(self.audio_path_file, self.transcript_path)
        self.word_list = [">"] + self.word_list
        # load model
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        target = 'vocals'

        # load model
        try:
            model_to_test = testx.load_model(target, model_path, device)
        except:
            model_path = "." + model_path
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

        # modify audio so that if there's a lot of zeros paddings on eitherside of the data, they will be removed
        from matplotlib import pyplot as plt
        start = 0
        end = -1
        for i in range(0, audio.shape[2]):
            if audio[0, 0, i].item() >= self.silence_threshold:
                start = max(0, i - 100)
                break
        for i in range(0, audio.shape[2]):
            if audio[0, 0, audio.shape[2]-1-i].item() > self.silence_threshold:
                end = min(audio.shape[2]-1-i + 100, audio.shape[2]-1)
                break
        audio = audio[:, :, start:end+1]
        start_off_set = 1/16000.0 * start

        # compute the alignment
        with torch.no_grad():
            vocals_estimate, alphas, scores = model_to_test((audio, phoneme_idx))
        optimal_path_scores = optimal_alignment_path(scores, mode='max_numpy', init=200)
        phoneme_onsets = compute_phoneme_onsets(optimal_path_scores, hop_length=nhop, sampling_rate=samplerate)
        phoneme_onsets = np.array(phoneme_onsets) + start_off_set

        # get the phoneme from the indexes
        temp_phoneme_intervals = []
        for i in range(1, phoneme_onsets.shape[0]-1):
            temp_phoneme_intervals.append([phoneme_onsets[i], phoneme_onsets[i+1]])
        temp_phoneme_list = data_parser.get_phonemes(phoneme_idx[0])[1:-1]
        temp_phoneme_list_full = phoneme_list_full[1:]

        # remove the spaces within words, but keep those between words
        i = 0 # pointer for the phoneme_list, increase by 1 each iteration
        j = 0 # pointer for the phoneme_list_full, increase by 1 each iteration, but increase by 2 at EOW
        while j < len(temp_phoneme_list_full)-1:
            # here we are assuming that temp_phoneme_list_full starts with no space
            if temp_phoneme_list_full[j+1] == ">":
                # if the next character is a space, then the intervals are merged
                self.phoneme_list.append(temp_phoneme_list[i])
                self.phoneme_intervals.append([temp_phoneme_intervals[i][0], temp_phoneme_intervals[i+1][1]])
                j = j + 2
                i = i + 2
            elif temp_phoneme_list_full[j+1] == "EOW":
                # if the next character is the end of word, then we just keep the current phoneme and interval
                self.phoneme_list.append(temp_phoneme_list[i])
                self.phoneme_intervals.append(temp_phoneme_intervals[i])
                j = j + 2
                i = i + 1
            elif temp_phoneme_list_full[j] == ">":
                # if we are at a space, that means we are that space between words, this will be kept.
                self.phoneme_list.append(temp_phoneme_list[i])
                self.phoneme_intervals.append(temp_phoneme_intervals[i])
                j = j + 1
                i = i + 1
            else:
                raise Exception("something is wrong with phoneme alignment")
        self.word_intervals = self.compute_word_alignment(phoneme_onsets, phoneme_list_full)
        self.phoneme_list = [">"] + self.phoneme_list
        self.phoneme_intervals = [[phoneme_onsets[0], phoneme_onsets[1]]] + self.phoneme_intervals
        print(self.phoneme_list)
        print(self.phoneme_intervals)
    def compute_self_viseme_alignment(self):

        dict_path = "./plla_tisvs/dicts"
        model_path = './plla_tisvs/trained_models/{}'.format("viseme")
        phoneme_dict_path = "cmu_word2cmu_phoneme_extra.pickle"

        # output_path = "E:/ten_videos/Child_in_time/Child_in_time_2"

        # initilize these variables
        self.phoneme_list = []
        self.phoneme_list_full = []
        self.phoneme_intervals = []
        self.word_list = []
        self.word_intervals = []

        # parse data
        try:
            data_parser = Custom_data_set(dict_path, phoneme_dict_path)
        except:
            dict_path = "." + dict_path
            data_parser = Custom_data_set(dict_path, phoneme_dict_path)
        audio, phoneme_idx, phoneme_list_full, self.word_list = data_parser.parse(self.audio_path_file, self.transcript_path, vocab="visemes")
        self.word_list = [">"] + self.word_list
        # load model
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        target = 'vocals'

        # load model
        try:
            model_to_test = testx.load_model(target, model_path, device)
        except:
            model_path = "." + model_path
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

        # modify audio so that if there's a lot of zeros paddings on eitherside of the data, they will be removed
        from matplotlib import pyplot as plt
        start = 0
        end = -1
        for i in range(0, audio.shape[2]):
            if audio[0, 0, i].item() >= self.silence_threshold:
                start = max(0, i - 100)
                break
        for i in range(0, audio.shape[2]):
            if audio[0, 0, audio.shape[2]-1-i].item() > self.silence_threshold:
                end = min(audio.shape[2]-1-i + 100, audio.shape[2]-1)
                break
        audio = audio[:, :, start:end+1]
        start_off_set = 1/16000.0 * start

        # compute the alignment
        with torch.no_grad():
            vocals_estimate, alphas, scores = model_to_test((audio, phoneme_idx))
        optimal_path_scores = optimal_alignment_path(scores, mode='max_numpy', init=200)
        phoneme_onsets = compute_phoneme_onsets(optimal_path_scores, hop_length=nhop, sampling_rate=samplerate)
        phoneme_onsets = np.array(phoneme_onsets) + start_off_set
        # get the phoneme from the indexes
        temp_phoneme_intervals = []
        for i in range(1, phoneme_onsets.shape[0]-1):
            temp_phoneme_intervals.append([phoneme_onsets[i], phoneme_onsets[i+1]])
        temp_phoneme_list = data_parser.get_visemes(phoneme_idx[0])[1:-1]
        temp_phoneme_list_full = phoneme_list_full[1:]

        # remove the spaces within words, but keep those between words
        i = 0 # pointer for the phoneme_list, increase by 1 each iteration
        j = 0 # pointer for the phoneme_list_full, increase by 1 each iteration, but increase by 2 at EOW
        while j < len(temp_phoneme_list_full)-1:
            # here we are assuming that temp_phoneme_list_full starts with no space
            if temp_phoneme_list_full[j+1] == ">":
                # if the next character is a space, then the intervals are merged
                self.phoneme_list.append(temp_phoneme_list[i])
                self.phoneme_intervals.append([temp_phoneme_intervals[i][0], temp_phoneme_intervals[i+1][1]])
                j = j + 2
                i = i + 2
            elif temp_phoneme_list_full[j+1] == "EOW":
                # if the next character is the end of word, then we just keep the current phoneme and interval
                self.phoneme_list.append(temp_phoneme_list[i])
                self.phoneme_intervals.append(temp_phoneme_intervals[i])
                j = j + 2
                i = i + 1
            elif temp_phoneme_list_full[j] == ">":
                # if we are at a space, that means we are that space between words, this will be kept.
                self.phoneme_list.append(temp_phoneme_list[i])
                self.phoneme_intervals.append(temp_phoneme_intervals[i])
                j = j + 1
                i = i + 1
            else:
                raise Exception("something is wrong with phoneme alignment")
        self.word_intervals = self.compute_word_alignment(phoneme_onsets, phoneme_list_full)
        self.phoneme_list = [">"] + self.phoneme_list
        self.phoneme_intervals = [[phoneme_onsets[0], phoneme_onsets[1]]] + self.phoneme_intervals
        print(self.phoneme_list)
        print(self.phoneme_intervals)
    def load_phoneme_textgrid(self, path):
        grid = textgrids.TextGrid(path)
        phoneme_list = []
        phoneme_intervals = []
        word_list = []
        word_intervals = []
        for i in range(0, len(grid["phones"])):
            phoneme_list.append(grid["phones"][i].text)
            phoneme_intervals.append([grid["phones"][i].xmin - grid.xmin, grid["phones"][i].xmax - grid.xmin])
        try:
            for i in range(0, len(grid["words"])):
                word_list.append(grid["words"][i].text)
                word_intervals.append([grid["words"][i].xmin - grid.xmin, grid["words"][i].xmax - grid.xmin])
        except:
            pass
        return phoneme_list, phoneme_intervals, word_list, word_intervals
    def write_textgrid(self, output_path, file_name):
        new_grid = textgrids.TextGrid()  # initialize new_textgrid object
        new_grid.xmin = 0
        new_grid.xmax = self.snd.xs()[-1]
        if len(self.phoneme_list) > 0:
            new_grid["phones"] = textgrids.Tier()
            for i in range(0, len(self.phoneme_list)):
                phoneme = self.phoneme_list[i]
                if phoneme == ">" or phoneme == "$":
                    phoneme = ">"
                interval = textgrids.Interval(phoneme, self.phoneme_intervals[i][0], self.phoneme_intervals[i][1])
                new_grid["phones"].append(interval)
        if len(self.word_list) > 0:
            new_grid["words"] = textgrids.Tier()
            for i in range(0, len(self.word_list)):
                interval = textgrids.Interval(self.word_list[i], self.word_intervals[i][0], self.word_intervals[i][1])
                new_grid["words"].append(interval)
        if len(self.vibrato_intervals) > 0:
            new_grid["vibrato"] = textgrids.Tier()
            for i in range(0, len(self.vibrato_intervals)):
                for vib_interval in self.vibrato_intervals[i]:
                    interval = textgrids.Interval("vibrato", vib_interval[0], vib_interval[1])
                    new_grid["vibrato"].append(interval)

        if len(self.pitch_slopes) > 0:
            new_grid["pitch_feature"] = textgrids.Tier()
            for i in range(0, len(self.pitch_slopes)):
                for j in range(0, len(self.pitch_intervals[i])):
                    pitch_interval = self.pitch_intervals[i][j]
                    interval = textgrids.Interval(int(self.pitch_slopes[i][j]), pitch_interval[0], pitch_interval[1])
                    new_grid["pitch_feature"].append(interval)

        if len(self.voice_quality_intervals) > 0:
            new_grid["voice_quality"] = textgrids.Tier()
            for i in range(0, len(self.voice_quality_intervals)):
                for j in range(0, len(self.voice_quality_intervals[i])):
                    interval = textgrids.Interval(self.voice_quality_lists[i][j], self.voice_quality_intervals[i][j][0], self.voice_quality_intervals[i][j][1])
                    new_grid["voice_quality"].append(interval)
        new_grid.write(os.path.join(output_path, file_name) + ".TextGrid")
    def __get_subarrays_indexes_from_time_interval(self, intervals, xs):
        # here i'm assuming that all interval in intervals have [t0, t1] where t0 < t1
        x_vals_low = [-1, -1]
        index_intervals = []


        # iterate through the intervals
        search_pointer = 0
        for interval in intervals:
            # find the start of the interval
            index_interval = [-1, -1]
            for i in range(search_pointer, xs.shape[0]):
                if xs[i] >= interval[0]:
                    index_interval[0] = i
                    search_pointer = i
                    break
            if interval[0] == interval[1]:
                index_interval[1] = search_pointer
            else:
                for i in range(search_pointer, xs.shape[0]):
                    if xs[i] >= interval[1]:
                        index_interval[1] = max(i - 1, 0)
                        search_pointer = max(i - 1, 0)
                        break
            index_intervals.append(index_interval)
        return index_intervals
    def get_subarrays_indexes_from_time_interval(self, intervals, xs):
        # here i'm assuming that all interval in intervals have [t0, t1] where t0 < t1
        x_vals_low = [-1, -1]
        index_intervals = []


        # iterate through the intervals
        search_pointer = 0
        for interval in intervals:
            # find the start of the interval
            index_interval = [-1, -1]
            for i in range(search_pointer, xs.shape[0]):
                if xs[i] >= interval[0]:
                    index_interval[0] = i
                    search_pointer = i
                    break
            if interval[0] == interval[1]:
                index_interval[1] = search_pointer
            else:
                for i in range(search_pointer, xs.shape[0]):
                    if xs[i] >= interval[1]:
                        index_interval[1] = max(i - 1, 0)
                        search_pointer = max(i - 1, 0)
                        break
            index_intervals.append(index_interval)
        return index_intervals
    def __compute_coarse_intervals(self, traits, intervals):
        new_intervals = []
        new_traits = []
        for i in range(0, len(intervals)):
            new_interval = []
            new_trait = []
            interval = intervals[i]
            trait = traits[i]
            if len(trait) > 1:
                prev_trait = trait[0]
                prev_index = 0
                for k in range(1, len(trait)):
                    if trait[k] == prev_trait and k == len(trait) - 1:
                        new_trait.append(prev_trait)
                        new_interval.append([interval[prev_index][0], interval[k][1]])
                    elif trait[k] == prev_trait:
                        continue
                    elif trait[k] != prev_trait:
                        new_trait.append(prev_trait)
                        new_interval.append([interval[prev_index][0], interval[k - 1][1]])
                        prev_trait = trait[k]
                new_traits.append(new_trait)
                new_intervals.append(new_interval)
            else:
                new_traits.append(trait)
                new_intervals.append(interval)
        return new_traits, new_intervals
class CMU_phonemes_dicts():
    def __init__(self):
        self.vocabs = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
                  'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH',
                  'UW', 'V', 'W', 'Y', 'Z', 'ZH'])
        self.vowels = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
                  'IH', 'IY', 'OW', 'OY', 'UH', 'UW', ])
        self.voiced = set(['M', 'N', "L", "NG"]).union(self.vowels)
        self.consonants = set(['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG',
                              'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'])
        self.consonants_no_jaw = self.consonants
        self.lip_closer = set(["B", "F", "M", "P", "S", "V"])
        self.lip_rounder = set(["B", "F", "M", "P", "V"])
        self.nasal_obtruents = set(['L', 'N', 'NG', 'T', 'D', 'G', 'K', 'F', 'V', 'M', 'B', 'P'])
        self.fricative = set(["S", "Z", "ZH", "SH", "CH", "F", "V", 'TH'])
        self.plosive = set(["P", "B", "D", "T", "K", "G"])
        self.lip_heavy = set(["W", "OW", "UW", "S", "Z", "Y", "JH", "OY"])
        self.sibilant = set(["S", "Z", "SH", "CH", "ZH"])
class JALI_visemes_dicts():
     def __init__(self):
        self.vowels = set(['Ih_pointer', 'Ee_pointer', 'Eh_pointer', 'Aa_pointer', 'U_pointer', 'Uh_pointer'
                           , 'Oo_pointer', 'Oh_pointer', 'Schwa_pointer', 'Eu_pointer', "Ah_pointer"])
        self.voiced = set(['Ih_pointer', 'Ee_pointer', 'Eh_pointer', 'Aa_pointer', 'U_pointer', 'Uh_pointer'
                           , 'Oo_pointer', 'Oh_pointer', 'Schwa_pointer', 'Eu_pointer', "Ah_pointer", "LNTD_pointer"])
        self.consonants_no_jaw = set(["Ya_pointer", "Ja_pointer", "Ra_pointer", "FVa_pointer", "LNTDa_pointer", "Ma_pointer", "BPa_pointer", "Wa_pointer", "Tha_pointer", "GKa_pointer"])
        self.consonants = set(["M_pointer", "BP_pointer", "JY_pointer", "Th_pointer", "ShChZh_pointer", "SZ_pointer", "GK_pointer", "LNTD_pointer", "R_pointer", "W_pointer", "FV_pointer"])
        self.lip_closer = set(["M_pointer", "BP_pointer", "FV_pointer", "SZ_pointer"])
        self.lip_rounder = set(["M_pointer", "BP_pointer", "FV_pointer"])
        self.vocabs = self.consonants.union(self.vowels).union(self.consonants_no_jaw)
        self.sibilant = set(["SZ_pointer", "ShChZh_pointer"])
        self.nasal_obtruents = set(["LNTD_pointer", "GK_pointer", "FV_pointer", "M_pointer", "BP_pointer"])
        self.fricative = set(["FV_pointer", "SZ_pointer", "ShChZh_pointer", "Th_pointer"])
        self.plosive = set(["BP_pointer", "LNTDa_pointer", "GK_pointer"])
        self.lip_heavy = set(["Oh_pointer", "W_pointer", "Wa_pointer", "U_pointer", "SZ_pointer", "JY_pointer",
                             "Ya_pointer", "Ja_pointer"])
        self.lip_rounder_to_no_jaw_dict = {"M_pointer":"Ma_pointer", "BP_pointer":"BPa_pointer", "FV_pointer":"FVa_pointer"}
class Viseme_curve():
    def __init__(self, v_list=None, v_pts=None):
        if v_list is None:
            self.viseme_list = []       # This should be a list of strings of CMU phonemes ["str"]
            self.viseme_ctrl_pts = []   # This should be a [[time:float, value:float, type:str]]
            self.pure_phoneme = []      # This should be a list of strings of CMU phonemes ["str"]
        else:
            self.viseme_list = [v_list]  # This should be a list of strings of CMU phonemes ["str"]
            self.viseme_ctrl_pts = [v_pts]  # This should be a [[time:float, value:float, type:str]]
            self.pure_phoneme = []  # This should be a list of strings of CMU phonemes ["str"]

    def new_pass(self):
        self.viseme_list.append([])
        self.viseme_ctrl_pts.append([])
        self.pure_phoneme.append([])
    def add(self, viseme, ctpts, phoneme):
        self.viseme_list[-1].append(viseme)
        self.viseme_ctrl_pts[-1].append(ctpts)
        self.pure_phoneme[-1].append(phoneme)
    def get(self, i, generation = -1):
        if i > len(self.viseme_list):
            return self.viseme_list[generation][-1], self.viseme_ctrl_pts[generation][-1], self.pure_phoneme[generation][-1]
        else:
            return self.viseme_list[generation][i], self.viseme_ctrl_pts[generation][i], self.pure_phoneme[generation][i]




