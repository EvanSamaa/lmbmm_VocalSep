import librosa
import json
import os
import torch
import numpy as np
if __name__ == "__main__":

    with open('./dicts/data_set_location.json') as f:
        dataset_path_dict = json.load(f)
    dataset_path = os.path.join(dataset_path_dict["dataset_root"], 'instrumentals')
    train_output_path = os.path.join(dataset_path_dict["dataset_root"], 'instrumentals/train/torch_snippets')
    val_output_path = os.path.join(dataset_path_dict["dataset_root"], 'instrumentals/val/torch_snippets')
    counter = 0;
    train_file_format = os.path.join(train_output_path, "{}.pt")
    val_file_format = os.path.join(val_output_path, "{}.pt")
    files = os.listdir(dataset_path)
    lengthOfSegment = 16000 * 8
    for file in files:
        if file[-3:] == "mp3":
            pass
        else:
            continue
        file_path = os.path.join(dataset_path, file)
        arr, sr = librosa.load(file_path, sr=16000)
        arr = (arr - arr.mean())/arr.std()
        for i in range(0, int(arr.shape[0]/lengthOfSegment)):
            min_val = i*lengthOfSegment
            max_val = min((i+1)*lengthOfSegment, arr.shape[0])
            segment = arr[min_val:max_val]
            audio_torch = torch.from_numpy(segment).type(torch.float32)
            audio_torch = audio_torch.repeat(2, 1)
            if counter < 1500:
                audio_path = train_file_format.format(counter)
            else:
                audio_path = val_file_format.format(counter)
            print(audio_path)
            torch.save(audio_torch, audio_path)
            counter = counter + 1