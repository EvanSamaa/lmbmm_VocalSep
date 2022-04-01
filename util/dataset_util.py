import musdb
import librosa as lb
import torch
import os
import pickle
import yaml
import json
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import soundfile as sf
import textgrids
from matplotlib import pyplot as plt
import parselmouth
import timit_utils as tu

TIMIT2CMU = {
    "iy":"iy",
    "ih":"ih",
    "eh":"eh",
    "ey":"ey",
    "ae":"ae",
    "aa":"aa",
    "aw":"aw",
    "ay":"ay",
    "ah":"ah",
    "ao":"ao",
    "oy":"oy",
    "ow":"ow",
    "uh":"uh",
    "uw":"uw",
    "ux":"uw",
    "er":"er",
    "ax":"ah",
    "ix":"ih",
    "axr":"er",
    "ax-h":"ah",
    "jh":"jh",
    "ch":"ch",
    "b":"b",
    "d":"d",
    "g":"g",
    "p":"p",
    "t":"t",
    "k":"k",
    "dx":"d",
    "s":"s",
    "sh":"sh",
    "z":"z",
    "zh":"zh",
    "f":"f",
    "th":"th",
    "v":"v",
    "dh":"dh",
    "m":"m",
    "n":"n",
    "ng":"ng",
    "em":["ah", "m"],
    "nx":["n", "er"],
    "en":["ah", "n"],
    "eng":["ah", "n"],
    "l":"l",
    "r":"r",
    "w":"w",
    "y":"y",
    "hh":"hh",
    "hv":"eh",
    "el":"ah",
    "bcl":"b",
    "dcl":"d",
    "gcl":"g",
    "pcl":"p",
    "tcl":"t",
    "kcl":"k",
    "q":"k",
    "pau":">",
    "epi":">",
    "h#": ">",

}
CMU_VOCAB = ['#', '$', '%', '>', '-', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
                  'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH',
                  'UW', 'V', 'W', 'Y', 'Z', 'ZH']

def get_timit_train_sentence(idx, timit_training_set):
    # the training set for this project comprises the first 4320 sentences of the TIMIT training partition
    # the persons are not sorted by dialect regions when accessed with .person_by_index, which ensures that all
    # dialect regions are represented in both the training and validation set
    person_idx = int(np.floor(idx / 10))
    person = timit_training_set.person_by_index(person_idx)
    sentence_idx = idx % 10
    sentence = person.sentence_by_index(sentence_idx)
    audio = sentence.raw_audio
    words = sentence.words_df.index.values
    phone_onset = sentence.phones_df['start'].values
    phone_offset = sentence.phones_df['end'].values
    phonemes = sentence.phones_df.index.values
    return audio, phonemes, phone_onset, phone_offset, words
def get_timit_val_sentence(idx, timit_training_set):
    # the validation set for this project comprises the last 300 sentences of the TIMIT training partition minus
    # the first two sentences per speaker (SA1, SA2) resulting in 240 utterance in total.
    # the persons are not sorted by dialect regions when accessed with .person_by_index, which ensures that all
    # dialect regions are represented in both the training and validation set
    person_idx = int(np.floor(idx / 8)) + 432
    person = timit_training_set.person_by_index(person_idx)
    sentence_idx = (idx % 8) + 2  # to ignore sentences 0 and 1 (SA1 and SA2), because they are also in training set
    sentence = person.sentence_by_index(sentence_idx)
    audio = sentence.raw_audio
    words = sentence.words_df.index.values
    phone_onset = sentence.phones_df['start'].values
    phone_offset = sentence.phones_df['end'].values
    phonemes = sentence.phones_df.index.values
    return audio, phonemes, phone_onset, phone_offset, words
def get_timit_test_sentence(idx, timit_test_set):

    person_idx = int(np.floor(idx / 8))
    person = timit_test_set.person_by_index(person_idx)
    sentence_idx = (idx % 8) + 2  # to ignore sentences 0 and 1 (SA1 and SA2), because they are also in training set
    sentence = person.sentence_by_index(sentence_idx)
    audio = sentence.raw_audio
    words = sentence.words_df.index.values
    phone_onset = sentence.phones_df['start'].values
    phone_offset = sentence.phones_df['end'].values
    phonemes = sentence.phones_df.index.values
    return audio, phonemes, phone_onset, phone_offset, words
def timit_sentence_to_cmu(timit_phone_list, timit_phone_start_list, timit_phone_end_list):
    cmu_phone_list, cmu_phone_start_list, cmu_phone_end_list = [[], [], []]
    for i in range(0, len(timit_phone_list)):
        timit_phone = timit_phone_list[i]
        cmu_phone = TIMIT2CMU[timit_phone]
        if timit_phone in ["em", "en", "eng"]:
            start = timit_phone_start_list[i]
            mid = timit_phone_end_list[i] - 2000
            end = timit_phone_end_list[i]
            if mid <= (start + end) / 2:
                mid = (start + end) / 2
            cmu_phone_list.append(cmu_phone[0])
            cmu_phone_start_list.append(start)
            cmu_phone_end_list.append(mid)

            cmu_phone_list.append(cmu_phone[1])
            cmu_phone_start_list.append(mid)
            cmu_phone_end_list.append(end)
        elif timit_phone in ["nx"]:
            start = timit_phone_start_list[i]
            mid = timit_phone_start_list[i] + 2000
            end = timit_phone_end_list[i]
            if mid >= (start + end) / 2:
                mid = (start + end) / 2
            cmu_phone_list.append(cmu_phone[0])
            cmu_phone_start_list.append(start)
            cmu_phone_end_list.append(mid)

            cmu_phone_list.append(cmu_phone[1])
            cmu_phone_start_list.append(mid)
            cmu_phone_end_list.append(end)
        else:
            cmu_phone_list.append(cmu_phone)
            cmu_phone_start_list.append(timit_phone_start_list[i])
            cmu_phone_end_list.append(timit_phone_end_list[i])
    return cmu_phone_list, cmu_phone_start_list, cmu_phone_end_list

def prepare_musdb():
    """
    This script reads the MUSDB lyrics annotation files, cuts the audio
    into snippets according to annotated lines, and
    saves audio and text files accordingly (both as torch files).

    Please note that when this file was written,
    the vocals category annotations were done with different
    letters than in the publicly available version of the MUSDB lyrics.
    The categories translate as follows: a-->n, b-->s, c-->d, d-->x (public format --> old format).
    This script can be used with the a, b, c, d annotation style but the
    annotations will be translated to the old format and the folder
    structure and other scripts use the old format as well.
    """
    # ignore warning about unsafe loaders in pyYAML 5.1 (used in musdb)
    # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
    yaml.warnings({'YAMLLoadWarning': False})

    with open('./location_dict.json') as f:
        dataset_path_dict = json.load(f)
    path_to_musdb = os.path.join(dataset_path_dict["dataset_root"], 'musdb18/')
    path_to_train_lyrics = os.path.join(dataset_path_dict["dataset_root"], 'Separation_data_sets/musdb18/train_lyrics/')
    path_to_test_lyrics = os.path.join(dataset_path_dict["dataset_root"], 'Separation_data_sets/musdb18/test_lyrics/')
    path_to_save_data = os.path.join(dataset_path_dict["dataset_root"], 'lmbmm_vocal_sep_data/musdb/')

    pickle_in = open('./plla_tisvs/dicts/char2idx.pickle', 'rb')
    char2idx = pickle.load(pickle_in)
    # char2idx = {}
    target_sr = 16000
    char2idx["-"] = 32
    # ------------------------------------------------------------------------------------------------------------------
    # make folder structure

    path = os.path.join(path_to_save_data, 'test', 'text')
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    for stem in ['vocals', 'mix', 'accompaniments']:
        for type in ['n', 'x', 's', 'd']:
            path = os.path.join(path_to_save_data, 'test', 'audio', stem, type)
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)

    path = os.path.join(path_to_save_data, 'val', 'text')
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    for stem in ['vocals', 'mix']:
        for type in ['n', 'x', 's', 'd']:
            path = os.path.join(path_to_save_data, 'val', 'audio', stem, type)
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)

    path = os.path.join(path_to_save_data, 'train', 'text')
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    for stem in ['vocals', 'accompaniment', 'drums', 'bass', 'other']:
        for type in ['n', 'x', 's', 'd']:
            path = os.path.join(path_to_save_data, 'train', 'audio', stem, type)
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path_to_save_data, 'train', 'audio', 'drums_12s'), exist_ok=True)
    os.makedirs(os.path.join(path_to_save_data, 'train', 'audio', 'bass_12s'), exist_ok=True)
    os.makedirs(os.path.join(path_to_save_data, 'train', 'audio', 'other_12s'), exist_ok=True)
    os.makedirs(os.path.join(path_to_save_data, 'train', 'audio', 'accompaniment_12s'), exist_ok=True)
    # ------------------------------------------------------------------------------------------------------------------

    musdb_corpus = musdb.DB(root=path_to_musdb)
    training_tracks = musdb_corpus.load_mus_tracks(subsets=['train'])
    test_tracks = musdb_corpus.load_mus_tracks(subsets=['test'])

    # validation set as open unmix but replace non english tracks by another track from the same artist:
    # replaced Fergessen - Nos Palpitants by Fergessen - The Wind
    # relplaced Meaxic - Take A Step by Meaxic - You Listen
    validation_tracks = ['Actions - One Minute Smile',
                         'Clara Berry And Wooldog - Waltz For My Victims',
                         'Johnny Lokke - Promises & Lies',
                         'Patrick Talbot - A Reason To Leave',
                         'Triviul - Angelsaint',
                         'Alexander Ross - Goodbye Bolero',
                         'Fergessen - The Wind',
                         'Leaf - Summerghost',
                         'Skelpolu - Human Mistakes',
                         'Young Griffo - Pennies',
                         'ANiMAL - Rockshow',
                         'James May - On The Line',
                         'Meaxic - You Listen',
                         'Traffic Experiment - Sirens']

    # -----------------------------------------------------------------------------------------------------------------
    # process MUSDB training partition and make training and validation files
    train_files_n = []
    train_files_s = []
    train_files_d = []
    train_files_x = []

    val_files_n = []
    val_files_s = []
    val_files_d = []
    val_files_x = []

    train_accompaniment_12s = []
    train_bass_12s = []
    train_drums_12s = []
    train_other_12s = []

    snippet_type_conversion = {'a': 'n', 'b': 's', 'c': 'd', 'd': 'x'}

    for track in training_tracks:

        track_name = track.name

        # make file name for audio and text files of current track
        file_name = track.name.split('-')
        file_name = file_name[0][0:6] + "_" + file_name[1][1:6]
        file_name = file_name.replace(" ", "_")

        # make boolean indicating whether current track is in validation set
        val_set_track = track_name in validation_tracks

        # -----------------------------------------------------------------------------------------------------------------
        # generate accompaniment snippets of 12 s length of all tracks in training partition
        if not val_set_track:
            for target in ['accompaniment', 'drums', 'bass', 'other']:
                accompaniment_audio = track.targets[target].audio
                accompaniment_audio_resampled = lb.core.resample(accompaniment_audio.T, track.rate, target_sr)
                acc_snippets = lb.util.frame(accompaniment_audio_resampled, frame_length=12 * target_sr,
                                             hop_length=12 * target_sr)

                number_of_snippets = acc_snippets.shape[-1]

                for counter in range(number_of_snippets):
                    # audio_torch has shape (2, ???) = (channels, samples)
                    audio_torch = torch.tensor(acc_snippets[:, :, counter]).type(torch.float32)
                    torch.save(audio_torch, os.path.join(path_to_save_data, 'train', 'audio', '{}_12s'.format(target),
                                                         file_name + '_{}.pt'.format(counter)))
                    if target == 'accompaniment':
                        train_accompaniment_12s.append(file_name + '_{}.pt'.format(counter))
                    elif target == 'drums':
                        train_drums_12s.append(file_name + '_{}.pt'.format(counter))
                    elif target == 'bass':
                        train_bass_12s.append(file_name + '_{}.pt'.format(counter))
                    elif target == 'other':
                        train_other_12s.append(file_name + '_{}.pt'.format(counter))
        # -----------------------------------------------------------------------------------------------------------------

        path_to_track_lyrics = os.path.join(path_to_train_lyrics, track_name + '.txt')

        # ignore files without lyrics annotations
        if not os.path.isfile(path_to_track_lyrics):
            print("No lyrics for", track, ", it was skipped")
            continue

        lyrics_file = open(path_to_track_lyrics)
        lyrics_lines = lyrics_file.readlines()

        vocals_audio = track.targets['vocals'].audio

        if val_set_track:
            other_audio = track.audio

            # resample
            acc_audio_resampled = lb.core.resample(other_audio.T, track.rate, target_sr)

            vocals_audio_resampled = lb.core.resample(vocals_audio.T, track.rate, target_sr)

            # go through lyrics lines and split audio as annotated
            for counter, line in enumerate(lyrics_lines):

                # ignore rejected lines
                if line[0] == '*':
                    continue

                annotations = line.split(' ', maxsplit=3)

                start_m = int(annotations[0].split(':')[0])  # start time minutes
                start_s = int(annotations[0].split(':')[1])  # start time seconds
                start_time = start_m * 60 + start_s  # start time in seconds

                end_m = int(annotations[1].split(':')[0])  # end time minutes
                end_s = int(annotations[1].split(':')[1])  # end time seconds
                end_time = end_m * 60 + end_s  # end time in seconds

                acc_audio_snippet = acc_audio_resampled[:, start_time * target_sr: end_time * target_sr]
                vocals_audio_snippet = vocals_audio_resampled[:, start_time * target_sr: end_time * target_sr]

                acc_audio_snippet_torch = torch.tensor(acc_audio_snippet).type(torch.float32)
                vocals_audio_snippet_torch = torch.tensor(vocals_audio_snippet).type(torch.float32)

                snippet_type = annotations[2]  # a, b, c, d

                snippet_type = snippet_type_conversion[snippet_type]  # change to old format n, s, d, x

                text = annotations[3].replace('\n', '').replace(' ', '>')
                # print(text)
                # Asadfjnna[2]
                text_idx = torch.tensor([char2idx[char] for char in text]).type(torch.float32)

                snippet_file_name = file_name + '_{}'.format(counter)

                partition = 'val'
                other = 'mix'

                # save audio
                path_to_save_vocals = os.path.join(path_to_save_data, partition, 'audio', 'vocals', snippet_type,
                                                   snippet_file_name)
                path_to_save_other = os.path.join(path_to_save_data, partition, 'audio', other, snippet_type,
                                                  snippet_file_name)
                torch.save(acc_audio_snippet_torch, path_to_save_other + '.pt')
                torch.save(vocals_audio_snippet_torch, path_to_save_vocals + '.pt')

                # save text
                path_to_save_text = os.path.join(path_to_save_data, partition, 'text', snippet_file_name + '.txt')
                path_to_save_text_idx = os.path.join(path_to_save_data, partition, 'text', snippet_file_name + '.pt')
                with open(path_to_save_text, 'w') as txt_file:
                    txt_file.write(text)
                    txt_file.close()
                torch.save(text_idx, path_to_save_text_idx)

                if snippet_type == 'n':
                    val_files_n.append('n/{}'.format(snippet_file_name))
                if snippet_type == 'x':
                    val_files_x.append('x/{}'.format(snippet_file_name))
                if snippet_type == 's':
                    val_files_s.append('s/{}'.format(snippet_file_name))
                if snippet_type == 'd':
                    val_files_d.append('d/{}'.format(snippet_file_name))

        # process training songs
        else:
            acc_audio = track.targets['accompaniment'].audio
            drums_audio = track.targets['drums'].audio
            bass_audio = track.targets['bass'].audio
            other_audio = track.targets['other'].audio

            # resample
            vocals_audio_resampled = lb.core.resample(vocals_audio.T, track.rate, target_sr)
            acc_audio_resampled = lb.core.resample(acc_audio.T, track.rate, target_sr)
            drums_audio_resampled = lb.core.resample(drums_audio.T, track.rate, target_sr)
            bass_audio_resampled = lb.core.resample(bass_audio.T, track.rate, target_sr)
            other_audio_resampled = lb.core.resample(other_audio.T, track.rate, target_sr)

            # go through lyrics lines and split audio as annotated
            for counter, line in enumerate(lyrics_lines):

                # ignore rejected lines
                if line[0] == '*' or line[0] == '-':
                    continue

                annotations = line.split(' ', maxsplit=3)

                start_m = int(annotations[0].split(':')[0])  # start time minutes
                start_s = int(annotations[0].split(':')[1])  # start time seconds
                start_time = start_m * 60 + start_s  # start time in seconds

                end_m = int(annotations[1].split(':')[0])  # end time minutes
                end_s = int(annotations[1].split(':')[1])  # end time seconds
                end_time = end_m * 60 + end_s  # end time in seconds

                acc_audio_snippet = acc_audio_resampled[:, start_time * target_sr: end_time * target_sr]
                vocals_audio_snippet = vocals_audio_resampled[:, start_time * target_sr: end_time * target_sr]
                drums_audio_snippet = drums_audio_resampled[:, start_time * target_sr: end_time * target_sr]
                bass_audio_snippet = bass_audio_resampled[:, start_time * target_sr: end_time * target_sr]
                other_audio_snippet = other_audio_resampled[:, start_time * target_sr: end_time * target_sr]

                acc_audio_snippet_torch = torch.tensor(acc_audio_snippet).type(torch.float32)
                vocals_audio_snippet_torch = torch.tensor(vocals_audio_snippet).type(torch.float32)
                drums_audio_snippet_torch = torch.tensor(drums_audio_snippet).type(torch.float32)
                bass_audio_snippet_torch = torch.tensor(bass_audio_snippet).type(torch.float32)
                other_audio_snippet_torch = torch.tensor(other_audio_snippet).type(torch.float32)

                snippet_type = annotations[2]  # a, b, c, d

                snippet_type = snippet_type_conversion[snippet_type]  # change to old format n, s, d, x

                text = annotations[3].replace('\n', '').replace(' ', '>')
                print(text)
                text_idx = torch.tensor([char2idx[char] for char in text]).type(torch.float32)

                snippet_file_name = file_name + '_{}'.format(counter)

                partition = 'train'
                other = 'accompaniments'

                # save audio
                path_to_save_vocals = os.path.join(path_to_save_data, partition, 'audio', 'vocals', snippet_type,
                                                   snippet_file_name)
                path_to_save_acc = os.path.join(path_to_save_data, partition, 'audio', 'accompaniment', snippet_type,
                                                snippet_file_name)
                path_to_save_drums = os.path.join(path_to_save_data, partition, 'audio', 'drums', snippet_type,
                                                  snippet_file_name)
                path_to_save_bass = os.path.join(path_to_save_data, partition, 'audio', 'bass', snippet_type,
                                                 snippet_file_name)
                path_to_save_other = os.path.join(path_to_save_data, partition, 'audio', 'other', snippet_type,
                                                  snippet_file_name)

                torch.save(acc_audio_snippet_torch, path_to_save_acc + '.pt')
                torch.save(vocals_audio_snippet_torch, path_to_save_vocals + '.pt')
                torch.save(drums_audio_snippet_torch, path_to_save_drums + '.pt')
                torch.save(bass_audio_snippet_torch, path_to_save_bass + '.pt')
                torch.save(other_audio_snippet_torch, path_to_save_other + '.pt')

                # save text
                path_to_save_text = os.path.join(path_to_save_data, partition, 'text', snippet_file_name + '.txt')
                path_to_save_text_idx = os.path.join(path_to_save_data, partition, 'text', snippet_file_name + '.pt')
                with open(path_to_save_text, 'w') as txt_file:
                    txt_file.write(text)
                    txt_file.close()
                torch.save(text_idx, path_to_save_text_idx)

                if snippet_type == 'n':
                    train_files_n.append('n/{}'.format(snippet_file_name))
                if snippet_type == 'x':
                    train_files_x.append('x/{}'.format(snippet_file_name))
                if snippet_type == 's':
                    train_files_s.append('s/{}'.format(snippet_file_name))
                if snippet_type == 'd':
                    train_files_d.append('d/{}'.format(snippet_file_name))

    # # save lists with file names
    pickle_out = open(os.path.join(path_to_save_data, "val", "val_files_n.pickle"), "wb")
    pickle.dump(val_files_n, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "val", "val_files_x.pickle"), "wb")
    pickle.dump(val_files_x, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "val", "val_files_s.pickle"), "wb")
    pickle.dump(val_files_s, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "val", "val_files_d.pickle"), "wb")
    pickle.dump(val_files_d, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(path_to_save_data, "train", "train_files_n.pickle"), "wb")
    pickle.dump(train_files_n, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "train", "train_files_x.pickle"), "wb")
    pickle.dump(train_files_x, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "train", "train_files_s.pickle"), "wb")
    pickle.dump(train_files_s, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "train", "train_files_d.pickle"), "wb")
    pickle.dump(train_files_d, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(path_to_save_data, "train", "train_accompaniments_12s.pickle"), "wb")
    pickle.dump(train_accompaniment_12s, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "train", "train_drums_12s.pickle"), "wb")
    pickle.dump(train_drums_12s, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "train", "train_bass_12s.pickle"), "wb")
    pickle.dump(train_bass_12s, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "train", "train_other_12s.pickle"), "wb")
    pickle.dump(train_other_12s, pickle_out)
    pickle_out.close()

    print("Train files n:", train_files_n)
    print("Train files x:", train_files_x)
    print("Train files s:", train_files_s)
    print("Train files d:", train_files_d)
    print("Val files n:", val_files_n)
    print("Val files x:", val_files_x)
    print("Val files s:", val_files_s)
    print("Val files d:", val_files_d)
    print("Train accompaniments 12s:", train_accompaniment_12s)
    # -----------------------------------------------------------------------------------------------------------------

    # process MUSDB test partition and make test files
    test_files_n = []
    test_files_s = []
    test_files_d = []
    test_files_x = []
    for track in test_tracks:

        track_name = track.name

        # make file name for audio and text files of current track
        file_name = track.name.split('-')
        file_name = file_name[0][0:6] + "_" + file_name[1][1:6]
        file_name = file_name.replace(" ", "_")

        path_to_track_lyrics = os.path.join(path_to_test_lyrics, track_name + '.txt')

        # ignore files without lyrics annotations
        if not os.path.isfile(path_to_track_lyrics):
            print("No lyrics for", track, ", it was skipped")
            continue

        lyrics_file = open(path_to_track_lyrics)
        lyrics_lines = lyrics_file.readlines()

        mix_audio = track.audio
        vocals_audio = track.targets['vocals'].audio
        accompaniment_audio = track.targets['accompaniment'].audio

        # resample
        mix_audio_resampled = lb.core.resample(mix_audio.T, track.rate, target_sr)
        vocals_audio_resampled = lb.core.resample(vocals_audio.T, track.rate, target_sr)
        accompaniment_audio_resampled = lb.core.resample(accompaniment_audio.T, track.rate, target_sr)

        # go through lyrics lines and split audio as annotated
        for counter, line in enumerate(lyrics_lines):

            # ignore rejected lines
            if line[0] == '*':
                continue

            annotations = line.split(' ', maxsplit=3)

            start_m = int(annotations[0].split(':')[0])  # start time minutes
            start_s = int(annotations[0].split(':')[1])  # start time seconds
            start_time = start_m * 60 + start_s  # start time in seconds

            end_m = int(annotations[1].split(':')[0])  # end time minutes
            end_s = int(annotations[1].split(':')[1])  # end time seconds
            end_time = end_m * 60 + end_s  # end time in seconds

            mix_audio_snippet = mix_audio_resampled[:, start_time * target_sr: end_time * target_sr]
            vocals_audio_snippet = vocals_audio_resampled[:, start_time * target_sr: end_time * target_sr]
            accompaniment_audio_snippet = accompaniment_audio_resampled[:, start_time * target_sr: end_time * target_sr]

            mix_audio_snippet_torch = torch.tensor(mix_audio_snippet).type(torch.float32)
            vocals_audio_snippet_torch = torch.tensor(vocals_audio_snippet).type(torch.float32)
            accompaniment_audio_snippet_torch = torch.tensor(accompaniment_audio_snippet).type(torch.float32)

            snippet_type = annotations[2]  # a, b, c, d

            snippet_type = snippet_type_conversion[snippet_type]  # change to old format n, s, d, x

            text = annotations[3].replace('\n', '').replace(' ', '>')
            text_idx = torch.tensor([char2idx[char] for char in text]).type(torch.float32)

            snippet_file_name = file_name + '_{}'.format(counter)

            # save audio
            path_to_save_vocals = os.path.join(path_to_save_data, 'test', 'audio', 'vocals', snippet_type,
                                               snippet_file_name)
            path_to_save_mix = os.path.join(path_to_save_data, 'test', 'audio', 'mix', snippet_type, snippet_file_name)
            path_to_save_acc = os.path.join(path_to_save_data, 'test', 'audio', 'accompaniments', snippet_type,
                                            snippet_file_name)

            torch.save(mix_audio_snippet_torch, path_to_save_mix + '.pt')
            torch.save(vocals_audio_snippet_torch, path_to_save_vocals + '.pt')
            torch.save(accompaniment_audio_snippet_torch, path_to_save_acc + '.pt')

            # save text
            path_to_save_text = os.path.join(path_to_save_data, 'test', 'text', snippet_file_name + '.txt')
            path_to_save_text_idx = os.path.join(path_to_save_data, 'test', 'text', snippet_file_name + '.pt')
            with open(path_to_save_text, 'w') as txt_file:
                txt_file.write(text)
                txt_file.close()
            torch.save(text_idx, path_to_save_text_idx)

            if snippet_type == 'n':
                test_files_n.append('n/{}'.format(snippet_file_name))
            if snippet_type == 'x':
                test_files_x.append('x/{}'.format(snippet_file_name))
            if snippet_type == 's':
                test_files_s.append('s/{}'.format(snippet_file_name))
            if snippet_type == 'd':
                test_files_d.append('d/{}'.format(snippet_file_name))

    # save lists with file names
    pickle_out = open(os.path.join(path_to_save_data, "test", "test_files_n.pickle"), "wb")
    pickle.dump(test_files_n, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "test", "test_files_x.pickle"), "wb")
    pickle.dump(test_files_x, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "test", "test_files_s.pickle"), "wb")
    pickle.dump(test_files_s, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_save_data, "test", "test_files_d.pickle"), "wb")
    pickle.dump(test_files_d, pickle_out)
    pickle_out.close()

    print("Test files n:", test_files_n)
    print("Test files x:", test_files_x)
    print("Test files s:", test_files_s)
    print("Test files d:", test_files_d)
def prepare_NUS():
    with open('./location_dict.json') as f:
        dataset_path_dict = json.load(f)
    path_to_dataset = os.path.join(dataset_path_dict["dataset_root"], 'Separation_data_sets/nus-smc-corpus_48/')
    path_to_save_data = os.path.join(dataset_path_dict["dataset_root"], 'lmbmm_vocal_sep_data/NUS/')
    train_set = [[], []]
    test_set = [[], []]
    speakers = os.listdir(path_to_dataset)
    for speaker in speakers:
        if speaker != "README.txt":
            # for each speaker
            sing_folder = os.path.join(os.path.join(path_to_dataset, speaker), "sing")
            file_for_folder = os.listdir(sing_folder)
            file_for_folder.sort()
            if speaker in ["SAMF", "VKOW", "ZHIY"]:
                for file_name in file_for_folder:
                    if file_name[0] != "." and file_name[-3:] == "wav":
                        test_set[0].append(os.path.join(sing_folder, file_name))
                    elif file_name[0] != "." and file_name[-3:] == "txt":
                        test_set[1].append(os.path.join(sing_folder, file_name))
            else:
                for file_name in file_for_folder:
                    if file_name[0] != "." and file_name[-3:] == "wav":
                        train_set[0].append(os.path.join(sing_folder, file_name))
                    elif file_name[0] != "." and file_name[-3:] == "txt":
                        train_set[1].append(os.path.join(sing_folder, file_name))
    train_set[0].sort(key=lambda student: student[-6:-4])
    train_set[1].sort(key=lambda student: student[-6:-4])
    test_set[0].sort(key=lambda student: student[-6:-4])
    test_set[1].sort(key=lambda student: student[-6:-4])

    test_set_location = os.path.join(path_to_save_data, "test")
    train_set_location = os.path.join(path_to_save_data, "train")

    try:
        os.mkdir(test_set_location)
        os.mkdir(train_set_location)
    except OSError:
        print("Directory already exist")

    # going through test set
    counter = 0
    # iterate through the audio files
    for i in range(0, len(test_set[0])):
        audio, fps = lb.load(test_set[0][i], sr=16000)
        lyric_content = open(test_set[1][i]).readlines()
        # z score normalization
        audio = (audio - audio.mean())/audio.std()/20.0
        # iterate through phonemes in the lyrics
        start = 0
        end = 0
        phoneme_timings = []
        for i in range(0, len(lyric_content)):
            # try:
            line = lyric_content[i].strip("\n")
            t_start, t_end, phone = line.split()
            # t_start = math.floor(float(t_start)*fps)
            # t_end = math.floor(float(t_end)*fps)
            if (phone == "sil" and len(phoneme_timings) >= 1) or (i==len(lyric_content)-1 and len(phoneme_timings) >= 1):
                if phone != "sil":
                    phoneme_timings.append([float(t_start), float(t_end), phone])
                    end = float(t_end)
                elif phone == "sil" and i==len(lyric_content)-1:
                    phoneme_timings.append([float(t_start), float(t_end), phone])
                    end = float(t_end)
                if end - start >= 4 or i==len(lyric_content)-1:
                    phonemme_transcript = textgrids.TextGrid()
                    phonemme_transcript.xmin = start
                    phonemme_transcript.xmax = end
                    phonemme_transcript["phones"] = textgrids.Tier()
                    for item in phoneme_timings:
                        if item[2] == "sil":
                            interval = textgrids.Interval(">", item[0], item[1])
                            phonemme_transcript["phones"].append(interval)
                        else:
                            interval = textgrids.Interval(item[2].upper(), item[0], item[1])
                            phonemme_transcript["phones"].append(interval)

                    ############### save audio and textgrid ###############
                    aud_file_save_path = os.path.join(test_set_location, "{}.pt".format(counter))
                    # aud_file_save_path = os.path.join(test_set_location, "{}.wav".format(counter))
                    # print(counter, end - start, audio[math.floor(start*fps):math.floor(end*fps)].shape[0]/fps, phoneme_timings[-1][1] - phoneme_timings[0][0])
                    aud_content = torch.from_numpy(audio[math.floor(start*fps):math.floor(end*fps)])
                    torch.save(aud_content, aud_file_save_path)
                    textgrid_save_path = os.path.join(test_set_location, "{}.TextGrid".format(counter))
                    phonemme_transcript.write(textgrid_save_path)
                    ############### manage tracking data
                    phoneme_timings = [[float(t_start), float(t_end), phone]]
                    start = float(t_start)
                    counter = counter + 1
            else:
                if phone != "sp":
                    phoneme_timings.append([float(t_start), float(t_end), phone])
                end = float(t_end)
    counter = 0
    start = 0
    end = 0
    for i in range(0, len(train_set[0])):
        audio, fps = lb.load(train_set[0][i], sr=16000)
        print(fps, train_set[0][i])
        lyric_content = open(train_set[1][i]).readlines()
        # z score normalization
        audio = (audio - audio.mean())/audio.std()/20.0
        # iterate through phonemes in the lyrics
        start = 0
        end = 0
        phoneme_timings = []
        for i in range(0, len(lyric_content)):
            # try:
            line = lyric_content[i].strip("\n")
            t_start, t_end, phone = line.split()
            # t_start = math.floor(float(t_start)*fps)
            # t_end = math.floor(float(t_end)*fps)
            if (phone == "sil" and len(phoneme_timings) >= 1) or (i==len(lyric_content)-1 and len(phoneme_timings) >= 1):
                if phone != "sil":
                    phoneme_timings.append([float(t_start), float(t_end), phone])
                    end = float(t_end)
                elif phone == "sil" and i==len(lyric_content)-1:
                    phoneme_timings.append([float(t_start), float(t_end), phone])
                    end = float(t_end)
                if end - start >= 4 or i==len(lyric_content)-1:
                    phonemme_transcript = textgrids.TextGrid()
                    phonemme_transcript.xmin = start
                    phonemme_transcript.xmax = end
                    phonemme_transcript["phones"] = textgrids.Tier()
                    for item in phoneme_timings:
                        if item[2] == "sil":
                            interval = textgrids.Interval(">", item[0], item[1])
                            phonemme_transcript["phones"].append(interval)
                        else:
                            interval = textgrids.Interval(item[2].upper(), item[0], item[1])
                            phonemme_transcript["phones"].append(interval)

                    ############### save audio and textgrid ###############
                    aud_file_save_path = os.path.join(train_set_location, "{}.pt".format(counter))
                    aud_content = torch.from_numpy(audio[math.floor(start*fps):math.floor(end*fps)])
                    torch.save(aud_content, aud_file_save_path)
                    textgrid_save_path = os.path.join(train_set_location, "{}.TextGrid".format(counter))
                    phonemme_transcript.write(textgrid_save_path)
                    ############### manage tracking data
                    phoneme_timings = [[float(t_start), float(t_end), phone]]
                    start = float(t_start)
                    counter = counter + 1
            else:
                if phone != "sp":
                    phoneme_timings.append([float(t_start), float(t_end), phone])
                end = float(t_end)
def analyze_NUS():
    with open('./location_dict.json') as f:
        dataset_path_dict = json.load(f)
    test_set_path = os.path.join(dataset_path_dict["dataset_root"], 'lmbmm_vocal_sep_data/NUS/test/')
    train_set_path = os.path.join(dataset_path_dict["dataset_root"], 'lmbmm_vocal_sep_data/NUS/train/')
    total_time_test = 0
    segment_lengths_test = []
    total_time_train = 0
    segment_lengths_train = []
    # test_files = os.listdir(test_set_path)
    for i in range(0, 249):
        aud = torch.load(os.path.join(test_set_path, "{}.pt".format(i)))
        total_time_test = total_time_test + aud.shape[0]
        segment_lengths_test.append(aud.shape[0]/16000.0)
    for i in range(0, 739):
        aud = torch.load(os.path.join(train_set_path, "{}.pt".format(i)))
        total_time_train = total_time_train + aud.shape[0]
        segment_lengths_train.append(aud.shape[0]/16000.0)
    print(total_time_test/16000.0)
    print(total_time_train/16000.0)
    import seaborn as sns
    sns.displot(segment_lengths_test, bins=50, kde=True)
    plt.show()
    sns.displot(segment_lengths_train, bins=50, kde=True)
    plt.show()
def prepare_timit():
    with open('location_dict.json') as f:
        dataset_path_dict = json.load(f)
    # insert path to your TIMIT corpus here
    print(dataset_path_dict["dataset_root"])
    corpus_path = os.path.join(dataset_path_dict["dataset_root"], "Separation_data_sets/timit/data/")
    corpus = tu.Corpus(corpus_path)
    # insert save path here
    save_path = os.path.join(dataset_path_dict["dataset_root"],
                                                        "lmbmm_vocal_sep_data/TIMIT/")
    train_path = os.path.join(save_path, "train")
    test_path = os.path.join(save_path, "test")
    # obtain corpus object using timit_util
    timit_training_set = corpus.train
    timit_test_set = corpus.test

    # gather training dataset
    counter = 0
    for idx in range(4320):
        audio, phonemes, phone_onset, phone_offset, words = get_timit_train_sentence(idx, timit_training_set)
        phonemes, phone_onset, phone_offset = timit_sentence_to_cmu(phonemes, phone_onset, phone_offset)
        phonemme_transcript = textgrids.TextGrid()
        phonemme_transcript.xmin = phone_onset[0]/16000.0
        phonemme_transcript.xmax = phone_offset[-1]/16000.0
        phonemme_transcript["phones"] = textgrids.Tier()
        audio = (audio - audio.mean())/audio.std()/20.0
        # remove the last space token
        for i in range(0, len(phonemes)):
            interval = textgrids.Interval(phonemes[i].upper(), phone_onset[i]/16000.0, phone_offset[i]/16000.0)
            phonemme_transcript["phones"].append(interval)

        aud_file_save_path = os.path.join(train_path, "{}.pt".format(counter))
        aud_content = torch.from_numpy(audio)
        torch.save(aud_content, aud_file_save_path)
        textgrid_save_path = os.path.join(train_path, "{}.TextGrid".format(counter))
        phonemme_transcript.write(textgrid_save_path)
        counter = counter + 1
    # validation sentences
    for idx in range(240):
        print(str(idx) + "\t is done")
        audio, phonemes, phone_onset, phone_offset, words = get_timit_val_sentence(idx, timit_training_set)
        phonemes, phone_onset, phone_offset = timit_sentence_to_cmu(phonemes, phone_onset, phone_offset)
        phonemme_transcript = textgrids.TextGrid()
        phonemme_transcript.xmin = phone_onset[0] / 16000.0
        phonemme_transcript.xmax = phone_offset[-1] / 16000.0
        phonemme_transcript["phones"] = textgrids.Tier()
        audio = (audio - audio.mean()) / audio.std() / 20.0
        # remove the last space token
        for i in range(0, len(phonemes)):
            # print("here: " + phonemes[i].upper() + "\t{}\t{}".format(phone_onset[i] / 16000.0, phone_offset[i] / 16000.0))
            interval = textgrids.Interval(phonemes[i].upper(), phone_onset[i] / 16000.0, phone_offset[i] / 16000.0)
            phonemme_transcript["phones"].append(interval)
        aud_file_save_path = os.path.join(train_path, "{}.pt".format(counter))
        aud_content = torch.from_numpy(audio)
        torch.save(aud_content, aud_file_save_path)
        textgrid_save_path = os.path.join(train_path, "{}.TextGrid".format(counter))
        phonemme_transcript.write(textgrid_save_path)
        counter = counter + 1
    counter = 0
    for idx in range(1344):
        print(str(idx) + "\t is done")
        audio, phonemes, phone_onset, phone_offset, words = get_timit_test_sentence(idx, timit_test_set)
        phonemes, phone_onset, phone_offset = timit_sentence_to_cmu(phonemes, phone_onset, phone_offset)
        phonemme_transcript = textgrids.TextGrid()
        phonemme_transcript.xmin = phone_onset[0] / 16000.0
        phonemme_transcript.xmax = phone_offset[-1] / 16000.0
        phonemme_transcript["phones"] = textgrids.Tier()
        audio = (audio - audio.mean()) / audio.std() / 20.0
        # remove the last space token
        for i in range(0, len(phonemes)):
            # print("here: " + phonemes[i].upper() + "\t{}\t{}".format(phone_onset[i] / 16000.0, phone_offset[i] / 16000.0))
            interval = textgrids.Interval(phonemes[i].upper(), phone_onset[i] / 16000.0, phone_offset[i] / 16000.0)
            phonemme_transcript["phones"].append(interval)
        aud_file_save_path = os.path.join(test_path, "{}.pt".format(counter))
        aud_content = torch.from_numpy(audio)
        torch.save(aud_content, aud_file_save_path)
        textgrid_save_path = os.path.join(test_path, "{}.TextGrid".format(counter))
        phonemme_transcript.write(textgrid_save_path)
        counter = counter + 1
def analyze_timit():
    with open('./location_dict.json') as f:
        dataset_path_dict = json.load(f)
    test_set_path = os.path.join(dataset_path_dict["dataset_root"], 'lmbmm_vocal_sep_data/TIMIT/test/')
    train_set_path = os.path.join(dataset_path_dict["dataset_root"], 'lmbmm_vocal_sep_data/TIMIT/train/')
    total_time_test = 0
    segment_lengths_test = []
    total_time_train = 0
    segment_lengths_train = []
    test_files = os.listdir(test_set_path)
    silences = []
    non_silences = []
    for i in range(0, 1343):
        aud = torch.load(os.path.join(test_set_path, "{}.pt".format(i)))
        grid = textgrids.TextGrid(os.path.join(test_set_path, "{}.TextGrid".format(i)))
        for i in range(0, len(grid["phones"])):
            if grid["phones"][i].text != ">":
                sil_interval = [grid["phones"][i].xmin, grid["phones"][i].xmax]
                sil_frames = [math.floor(sil_interval[0]*16000), math.floor(sil_interval[1]*16000)]
                silences.append((aud[sil_frames[0]:sil_frames[1]]**2).mean().detach().numpy().sum())
            else:
                sil_interval = [grid["phones"][i].xmin, grid["phones"][i].xmax]
                sil_frames = [math.floor(sil_interval[0] * 16000), math.floor(sil_interval[1] * 16000)]
                non_silences.append((aud[sil_frames[0]:sil_frames[1]] ** 2).mean().detach().numpy().sum())
        total_time_test = total_time_test + aud.shape[0]
        segment_lengths_test.append(aud.shape[0]/16000.0)
    for i in range(0, 4559):
        aud = torch.load(os.path.join(train_set_path, "{}.pt".format(i)))
        total_time_train = total_time_train + aud.shape[0]
        segment_lengths_train.append(aud.shape[0]/16000.0)
        for i in range(0, len(grid["phones"])):
            if grid["phones"][i].text != ">":
                sil_interval = [grid["phones"][i].xmin, grid["phones"][i].xmax]
                sil_frames = [math.floor(sil_interval[0]*16000), math.floor(sil_interval[1]*16000)]
                silences.append((aud[sil_frames[0]:sil_frames[1]]**2).mean().detach().numpy().sum())
            else:
                sil_interval = [grid["phones"][i].xmin, grid["phones"][i].xmax]
                sil_frames = [math.floor(sil_interval[0] * 16000), math.floor(sil_interval[1] * 16000)]
                non_silences.append((aud[sil_frames[0]:sil_frames[1]] ** 2).mean().detach().numpy().sum())
    print(total_time_test/16000.0)
    print(total_time_train/16000.0)
    import seaborn as sns

    # sns.displot(test_silences, bins=50, kde=True)

    plt.show()
    sns.displot(segment_lengths_test, bins=50, kde=True)
    # plt.show()
    sns.displot(segment_lengths_train, bins=50, kde=True)
    plt.show()

def prepare_NUS_landmarks():
    with open('./location_dict.json') as f:
        dataset_path_dict = json.load(f)
    path_to_dataset = os.path.join(dataset_path_dict["dataset_root"], 'lmbmm_vocal_sep_data/NUS/')
    # for test set
    for i in range(0, 270):
        filePath = os.path.join(path_to_dataset, "test_landmarks_raw/{}_raw.json".format(i))
        savePath = os.path.join(path_to_dataset, "test_landmarks_raw/{}_processed.pt".format(i))
        with open(filePath) as f:
            test_file = json.load(f)
        keys = list(test_file["landmarks"].keys())
        keys.sort()


        N = len(keys)
        T = len(test_file["landmarks"][keys[0]])
        landmark_array = np.zeros([T, N, 3])
        # landmark_array_t = np.zeros([N, 3])
        for tt in range(0, T):
            for n in range(0, N):
                landmark_array[tt][n] = np.array(test_file["landmarks"][keys[n]][tt])
                # landmark_array_t[n] = np.array(test_file["landmarks"][keys[n]][tt])
            landmark_array[tt] = landmark_array[tt] - landmark_array[tt].mean(axis=0, keepdims=True)
            landmark_array[tt] = landmark_array[tt]/np.sqrt(np.square(landmark_array[tt]).sum(axis=1)).mean()

            # plt.scatter(landmark_array_t[:, 0], landmark_array_t[:, 1])
            # if tt == T-1:
            #     plt.show()
        landmark_array = torch.from_numpy(landmark_array).type(torch.FloatTensor)
        torch.save(landmark_array, savePath)

        print(i, "test")

    for i in range(0, 854):
        filePath = os.path.join(path_to_dataset, "train_landmarks_raw/{}_raw.json".format(i))
        savePath = os.path.join(path_to_dataset, "train_landmarks_raw/{}_processed.pt".format(i))
        with open(filePath) as f:
            test_file = json.load(f)
        keys = list(test_file["landmarks"].keys())
        keys.sort()
        minn = np.ceil(test_file["t_min"]) * 24
        maxx = np.ceil(test_file["t_max"]) * 24
        N = len(keys)
        T = len(test_file["landmarks"][keys[0]])
        landmark_array = np.zeros([T, N, 3])
        # landmark_array_t = np.zeros([N, 3])
        for tt in range(0, T):
            for n in range(0, N):
                landmark_array[tt][n] = np.array(test_file["landmarks"][keys[n]][tt])
            landmark_array[tt] = landmark_array[tt] - landmark_array[tt].mean(axis=0, keepdims=True)
            landmark_array[tt] = landmark_array[tt]/np.sqrt(np.square(landmark_array[tt]).sum(axis=1)).mean()
        landmark_array = torch.from_numpy(landmark_array).type(torch.FloatTensor)
        torch.save(landmark_array, savePath)
        # plt.scatter(landmark_array_t[:,0], landmark_array_t[:, 1])
        # plt.show()
        print(i, "train")


def prepare_instrumental():
    with open('location_dict.json') as f:
        dataset_path_dict = json.load(f)
    dataset_path = os.path.join(dataset_path_dict["dataset_root"], 'Separation_data_sets/instrumentals')
    output_path = os.path.join(dataset_path_dict["dataset_root"], 'lmbmm_vocal_sep_data/INSTRUMENT/data/')
    # val_output_path = os.path.join(dataset_path_dict["dataset_root"], 'INSTRUMENT/test/')
    counter = 0
    output_file_format = os.path.join(output_path, "{}.pt")
    files = os.listdir(dataset_path)
    lengthOfSegment = 16000 * 12
    total_time = 0
    for file in files:
        if file[-3:] == "mp3":
            pass
        else:
            continue
        file_path = os.path.join(dataset_path, file)
        arr, sr = lb.load(file_path, sr=16000)
        arr = (arr - arr.mean())/arr.std()/20.0
        total_time = total_time + arr.shape[0]
        for i in range(0, int(arr.shape[0]/lengthOfSegment)):
            min_val = i*lengthOfSegment
            max_val = min((i+1)*lengthOfSegment, arr.shape[0])
            if max_val - min_val < lengthOfSegment:
                continue
            segment = arr[min_val:max_val]
            audio_torch = torch.from_numpy(segment).type(torch.float32)
            audio_torch = audio_torch.repeat(2, 1)
            audio_path = output_file_format.format(counter)
            print(audio_path)
            torch.save(audio_torch, audio_path)
            counter = counter + 1
        print("completed " + file +  ", so far we have parsed {} seconds of music.".format(total_time))
def prep_raw_landmarks():
    # this can only be done after the keypoint data are extracted from maya
    with open('./location_dict.json') as f:
        dataset_path_dict = json.load(f)
    path_to_test = os.path.join(dataset_path_dict["dataset_root"], 'lmbmm_vocal_sep_data/NUS/test_landmarks/{}_raw.json')
    with open(path_to_test.format(0)) as f:
        lm = json.load(f)
    minn = np.floor(lm["t_min"])*24
    maxx = np.ceil(lm["t_max"])*24
    lm_indexes = sorted(list(lm["landmarks"].keys()), key=lambda x: int(x))
    # since we will be storing the landmarks in an array, we need to convert the landmark indices to array indices
    # we therefore construct two dictionaries to keep track of the conversion
    index_to_landmark = {}
    landmark_to_index = {}
    for i in range(0, len(lm_indexes)):
        landmark_to_index[lm_indexes[i]] = i
        index_to_landmark[i] = lm_indexes[i]

    # this visualizes it
    for i in range(0, 4):
        # for each file load the json file
        lm_dict = {}
        with open(path_to_test.format(i)) as f:
            lm_dict = json.load(f)
        t = 0
        # plot the subsequent points
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for lm in lm_indexes:
            ax.scatter(lm_dict["landmarks"][lm][t][0], lm_dict["landmarks"][lm][t][1], lm_dict["landmarks"][lm][t][2])

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

        plt.show()
        A[2]





