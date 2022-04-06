import torch
import model
import json
from matplotlib import pyplot as plt
import numpy as np
def qualitative_eval_pipeline(fileName):

    return 0
def plotLandmarks():
    path = "F:\\MASC\\lmbmm_vocal_sep_data\\NUS\\test_landmarks_raw\\171_processed.pt"
    arr = torch.load(path)
    arr = arr.numpy()
    X = arr[0, :, 0]
    Y = arr[0, :, 1]
    plt.scatter(X, Y)
    annotations = list(range(0, arr.shape[1]))
    for i, label in enumerate(annotations):
        plt.annotate(label, (X[i], Y[i]))
    plt.show()
if __name__ == "__main__":
    plotLandmarks()
    A[2]

    device = "cpu"
    target_model_path = "trained_models/unmix_toy_NUS_ONLY/vocal.chkpnt"
    checkpoint = torch.load(target_model_path, map_location=device)
    with open("training_specs/toy_example_unmix.json") as f:
        specs = json.load(f)
    # input_specs
    model_of_concern = model.OpenUnmix(sample_rate=specs["sample_rate"], n_fft=specs["n_fft"], n_hop=specs["n_hop"], input_is_spectrogram=False)
    model_of_concern.load_state_dict(checkpoint['state_dict'])