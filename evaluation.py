import torch
import model
import json

def pipeline(fileName):
    return 0


if __name__ == "__main__":
    device = "cpu"
    target_model_path = "trained_models/unmix_toy_NUS_ONLY/vocal.chkpnt"
    checkpoint = torch.load(target_model_path, map_location=device)
    with open("training_specs/toy_example_unmix.json") as f:
        specs = json.load(f)
    # input_specs
    model_of_concern = model.OpenUnmix(sample_rate=specs["sample_rate"], n_fft=specs["n_fft"], n_hop=specs["n_hop"], input_is_spectrogram=False)
    model_of_concern.load_state_dict(checkpoint['state_dict'])