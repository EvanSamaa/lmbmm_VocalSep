import torch
import model as model_classes
import json
from matplotlib import pyplot as plt
import numpy as np
import norbert
from scipy.signal import istft as sisft
import os
from pathlib import Path
from util.data_loader import *
import museval
import pandas as pd
import torch.nn.functional as F

def istft(X, rate=16000, n_fft=4096, n_hopsize=1024):
    t, audio = sisft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio
def evaluate(references, estimates, output_dir, track_name, sample_rate, win=1.0, hop=1.0, mode='v4'):

    """
    Compute the BSS_eval metrics as well as PES and EPS. It is following the design concept of museval.eval_mus_track
    :param references: dict of reference sources {target_name: signal}, signal has shape: (nb_timesteps, np_channels)
    :param estimates: dict of user estimates {target_name: signal}, signal has shape: (nb_timesteps, np_channels)
    :param output_dir: path to output directory used to save evaluation results
    :param track_name: name that is assigned to TrackStore object for evaluated track
    :param win: evaluation window length in seconds, default 1
    :param hop: evaluation window hop length in second, default 1
    :param sample_rate: sample rate of test tracks (should be same as rate the model has been trained on)
    :param mode: BSSEval version, default to `v4`
    :return:
        bss_eval_data: museval.TrackStore object containing bss_eval evaluation scores
        silent_frames_data: Pandas data frame containing EPS and PES scores
    """

    eval_targets = list(estimates.keys())

    estimates_list = []
    references_list = []
    for target in eval_targets:
        estimates_list.append(estimates[target].T)
        references_list.append(references[target])

    # eval bass_eval and EPS, PES metrics
    # save in TrackStore object
    bss_eval_data = museval.TrackStore(win=win, hop=hop, track_name=track_name)

    # skip examples with a silent source because BSSeval metrics are not defined in this case
    skip = False
    for target in eval_targets:
        reference_energy = np.sum(references[target]**2)
        estimate_energy = np.sum(estimates[target]**2)
        if reference_energy == 0 or estimate_energy == 0:
            skip = True
            SDR = ISR = SIR = SAR = (np.ones((1,)) * (-np.inf), np.ones((1,)) * (-np.inf))
            print("skip {}, {} source is all zero".format(track_name, target))

    if not skip:

        SDR, ISR, SIR, SAR = museval.evaluate(
            references_list,
            estimates_list,
            win=int(win * sample_rate),
            hop=int(hop * sample_rate),
            mode=mode,
            padding=True
        )


    # iterate over all targets
    for i, target in enumerate(eval_targets):
        values = {
            "SDR": SDR[i].tolist(),
            "SIR": SIR[i].tolist(),
            "ISR": ISR[i].tolist(),
            "SAR": SAR[i].tolist(),
        }

        bss_eval_data.add_target(
            target_name=target,
            values=values
        )

    # save evaluation results if output directory is defined
    if output_dir:
        # validate against the schema
        bss_eval_data.validate()

        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(
                    os.path.join(output_dir, track_name.replace('/', '_')) + '.json', 'w+'
            ) as f:
                f.write(bss_eval_data.json)
        except (IOError):
            pass

    return bss_eval_data
def separate_and_evaluate2(
    track,
    targets,
    model,
    niter,
    alpha,
    softmask,
    output_dir,
    eval_dir,
    samplerate,
    device='cpu',
    args=None,
    index=0,
):
    mix = track[0]
    true_vocals = track[1]
    true_accompaniment = mix - true_vocals
    text = track[2].unsqueeze(dim=0)
    track_name = "test_track_{}".format(index)

    mix_numpy = mix.numpy().T

    inputs = (mix, text)

    estimates = separate(
        inputs=inputs,
        targets=targets,
        model=model,
        niter=niter,
        alpha=alpha,
        softmask=softmask,
        device=device,
        args=args,
    )

    # if output_dir:
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     # make another script that reassembles kind of whole tracks of snippets
    #     sf.write(os.path.join(output_dir, track_name.replace('/', '_') + '.wav'), estimates['vocals'], samplerate)
    references = {'vocals': true_vocals.numpy().T, 'instrumental': true_accompaniment.numpy().T}
    bss_eval_scores = evaluate(references=references,
                                                     estimates=estimates,
                                                     output_dir=eval_dir,
                                                     track_name=track_name,
                                                     sample_rate=samplerate)

    return bss_eval_scores

def separate(
    inputs,
    targets,
    model,
    niter=1, softmask=False, alpha=1.0,
    residual_model=False, device='cpu', args=None):
    """
    Performing the separation on audio input

    Parameters
    ----------
    inputs: tuple of mixture and side info: (np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)], torch.tensor)
        mixture audio
        (comment by Kilian: it looks like the expected np.ndarray shape is actually (nb_timesteps, nb_channels),
        the torch tensor audio_torch then gets the shape (nb_samples, nb_channels, nb_timesteps))

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.

    """
    mix = torch.unsqueeze(inputs[0], axis = 0)
    mix_numpy = mix.numpy().T
    text = inputs[1]
    # text = torch.unsqueeze(inputs[1], axis = 0)
    # convert numpy audio to torch
    V = []
    source_names = []
    with torch.no_grad():
        # out = model((mix, text))
        out = model((mix, text))
        # if type(out) == list:
        #     out = out[0]
        Vj = out.cpu().detach().numpy()
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ...])  # remove sample dim
        source_names += ["vocal"]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    X = model.stft(mix).detach().cpu().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1]*1j
    X = X[0].transpose(2, 1, 0)

    Y = norbert.wiener(V, X.astype(np.complex128), 1,
                       use_softmask=softmask)
    # print(X.shape, Y.shape)
    audio_hat = istft(
        Y[..., 0].T,
        n_fft=model.stft.n_fft,
        n_hopsize=model.stft.n_hop
    )
    instrumental_hat = istft(X.T, n_fft=model.stft.n_fft, n_hopsize=model.stft.n_hop) - audio_hat
    estimates = {"vocals":audio_hat, "instrumental":instrumental_hat}
    return estimates


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

def compute_BSS_Metrics():
    with open("training_specs/toy_example_unmix.json") as f:
        specs_unmix = json.load(f)
    unmix = model_classes.OpenUnmix(sample_rate=specs_unmix["sample_rate"], n_fft=specs_unmix["n_fft"],
                            n_hop=specs_unmix["n_hop"], input_is_spectrogram=False)
    with open("training_specs/toy_example_naive_landmark_only_unmix.json") as f:
        specs_unmix_AI = json.load(f)
    unmix_AI = model_classes.OpenUnmixWithLandmarks(sample_rate=specs_unmix_AI["sample_rate"], landmarkCount=38)
    with open("training_specs/toy_example_naive_landmark_only_unmix_duo_objective.json") as f:
        specs_unmiM_AIO = json.load(f)
    unmiM_AIO = model_classes.OpenUnmixWithLandmarks3(sample_rate=specs_unmiM_AIO["sample_rate"], landmarkCount=38)
    with open("training_specs/toy_example_only_unmix_duo_objective_shallowmodel.json") as f:
        specs_unmiM_AO = json.load(f)
    unmiM_AO = model_classes.OpenUnmixWithLandmarks5(sample_rate=specs_unmiM_AO["sample_rate"], landmarkCount=38)

    specss = [specs_unmix, specs_unmix_AI, specs_unmiM_AO, specs_unmiM_AIO]
    models = [unmix, unmix_AI, unmiM_AO, unmiM_AIO]
    for i in range(0, 2):
        model = models[i]
        specs = specss[i]
        # location = spec["pre_train_location"]
        if specs["pre_train_location"] != "":
            try:
                model_path = Path(os.path.join('trained_models/', specs["pre_train_location"])).expanduser()
                with open(Path(os.path.join(model_path, "vocal" + '.json')), 'r') as stream:
                    results = json.load(stream)

                target_model_path = Path(model_path, "vocal" + ".chkpnt")
                use_cuda = torch.cuda.is_available()
                device = torch.device("cuda" if use_cuda else "cpu")
                checkpoint = torch.load(target_model_path, device)
                print("loaded checkpoint")
                model.load_state_dict(checkpoint['state_dict'])
                print("loaded model")
            except:
                print("model loading unsuccessful")
        # model is now loaded, so here loads the dataset
        if specs["dataset"] == "TIMIT":
            valid_dataset = TIMITMusicTest(None, fixed_length=True, size=500, mono=True)
        elif specs["dataset"] == "NUS":
            valid_dataset = NUSMusicTest(None, fixed_length=True, size=500, mono=True, random_song=False)
        elif specs["dataset"] == "NUS_landmark":
            valid_dataset = NUSMusicTest("landmarks", fixed_length=True, mono=True, landmarkNoise=0, random_song=True)
        elif specs["dataset"] == "NUS_landmark_sparse":
            valid_dataset = NUSMusicTest("landmarks_sparse", fixed_length=True, mono=True, landmarkNoise=0)
        results = museval.EvalStore()
        for idx in tqdm.tqdm(range(len(valid_dataset))):
            try:
                track = valid_dataset[idx]
                # track[0].shape = torch.Size([2, 192000])
                # track[1].shape = torch.Size([2, 192000])
                # track[2].shape = torch.Size([288, 76])
                bss_eval_scores = separate_and_evaluate2(
                    track,
                    targets=["vocals", "instrumental"],
                    model=model,
                    niter=None,
                    alpha=None,
                    softmask=None,
                    output_dir=None,
                    eval_dir=None,
                    device=device,
                    samplerate=16000,
                    index=idx
                )
                print(bss_eval_scores)
                results.add_track(bss_eval_scores)
            except:
                print("failed at track {}".format(idx))
        method = museval.MethodStore()
        method.add_evalstore(results, specs["name"])
        method.save('trained_models/modelEval/{}.pkl'.format(specs["name"]))
        # silent_frames_results = silent_frames_results.append(silent_frames_scores, ignore_index=True)
def compute_testing_loss():
    with open("training_specs/toy_example_unmix.json") as f:
        specs_unmix = json.load(f)
    unmix = model_classes.OpenUnmix(sample_rate=specs_unmix["sample_rate"], n_fft=specs_unmix["n_fft"],
                            n_hop=specs_unmix["n_hop"], input_is_spectrogram=False)
    with open("training_specs/toy_example_naive_landmark_only_unmix.json") as f:
        specs_unmix_AI = json.load(f)
    unmix_AI = model_classes.OpenUnmixWithLandmarks(sample_rate=specs_unmix_AI["sample_rate"], landmarkCount=38)
    with open("training_specs/toy_example_naive_landmark_only_unmix_duo_objective.json") as f:
        specs_unmiM_AIO = json.load(f)
    unmiM_AIO = model_classes.OpenUnmixWithLandmarks3(sample_rate=specs_unmiM_AIO["sample_rate"], landmarkCount=38)
    with open("training_specs/toy_example_only_unmix_duo_objective_shallowmodel.json") as f:
        specs_unmiM_AO = json.load(f)
    unmiM_AO = model_classes.OpenUnmixWithLandmarks5(sample_rate=specs_unmiM_AO["sample_rate"], landmarkCount=38)

    specss = [specs_unmix, specs_unmix_AI, specs_unmiM_AO, specs_unmiM_AIO]
    models = [unmix, unmix_AI, unmiM_AO, unmiM_AIO]
    for i in range(2, 4):
        model = models[i]
        specs = specss[i]
        # location = spec["pre_train_location"]
        if specs["pre_train_location"] != "":
            try:
                model_path = Path(os.path.join('trained_models/', specs["pre_train_location"])).expanduser()
                with open(Path(os.path.join(model_path, "vocal" + '.json')), 'r') as stream:
                    results = json.load(stream)

                target_model_path = Path(model_path, "vocal" + ".chkpnt")
                use_cuda = torch.cuda.is_available()
                device = torch.device("cuda" if use_cuda else "cpu")
                checkpoint = torch.load(target_model_path, device)
                print("loaded checkpoint")
                model.load_state_dict(checkpoint['state_dict'])
                print("loaded model")
            except:
                print("model loading unsuccessful")
        # model is now loaded, so here loads the dataset
        if specs["dataset"] == "TIMIT":
            valid_dataset = TIMITMusicTest(None, fixed_length=True, size=500, mono=True)
        elif specs["dataset"] == "NUS":
            valid_dataset = NUSMusicTest(None, fixed_length=True, size=500, mono=True, random_song=False)
        elif specs["dataset"] == "NUS_landmark":
            valid_dataset = NUSMusicTest("landmarks", fixed_length=True, mono=True, landmarkNoise=0, random_song=True)
        elif specs["dataset"] == "NUS_landmark_sparse":
            valid_dataset = NUSMusicTest("landmarks_sparse", fixed_length=True, mono=True, landmarkNoise=0)
        valid_sampler = torch.utils.data.DataLoader(
            valid_dataset, batch_size=1, shuffle=True, drop_last=True,
        )
        results = []
        model.eval()
        model.stft.center = True
        with torch.no_grad():
            for idx in tqdm.tqdm(range(len(valid_dataset))):
                try:
                    data = valid_dataset[idx]
                    x = torch.unsqueeze(data[0], axis=0)  # mix
                    y = torch.unsqueeze(data[1], axis=0)  # vocals
                    z = torch.unsqueeze(data[2], axis=0)  # text
                    x, y, z = x.to(device), y.to(device), z.to(device)
                    # if args.alignment_from:
                    #     inputs = (x, z, data[3].to(device))  # add attention weight to input
                    # else:
                    inputs = (x, z)
                    Y_hat, L_hat = model(inputs)
                    loss_fn2 = torch.nn.MSELoss()
                    L = z.permute((0, 2, 1))
                    # landmarks = [Batch, L*2, new_T]
                    L = F.interpolate(L, L_hat.shape[1])
                    # landmarks = [new_T, Batch, L*2]
                    L = L.permute((0, 2, 1))

                    Y = model.transform(y)
                    loss_fn = torch.nn.L1Loss(
                        reduction='sum')  # in sms project, the loss is defined before looping over epochs
                    loss = loss_fn(Y_hat, Y)
                    results.append(loss_fn2(L_hat, L).numpy())
                except:
                    print("skipped {}".format(idx))
        np.save('trained_models/modelEval/MSE_{}.npy'.format(specs["name"]), np.array(results))
        # method = museval.MethodStore()
        # method.add_evalstore(results, specs["name"])
        # method.save('trained_models/modelEval/{}.pkl'.format(specs["name"]))
        # silent_frames_results = silent_frames_results.append(silent_frames_scores, ignore_index=True)
if __name__ == "__main__":
    compute_BSS_Metrics()
    A[2]
    # m1 = np.load("trained_models/modelEval/MSE_lst_landmark_unmix_only_toy_NUS_ONLY_DUO_objective.npy")
    # m2 = np.load("trained_models/modelEval/MSE_lunmix_only_toy_NUS_ONLY_DUO_objective_shallow_model.npy")
    # plt.boxplot([m1, m2], labels=["unmix-AIO", "unmix-AO"])
    # plt.ylabel("MSE loss with target landmark")
    #
    # # show plot
    # plt.show()
    # print(m1.mean())
    # print(m2.mean())
    #
    # # compute_testing_loss()
    # A[2]
    evalPaths = "trained_models/modelEval/{}.pkl"
    with open("training_specs/toy_example_unmix.json") as f:
        specs_unmix = json.load(f)
    unmix = model_classes.OpenUnmix(sample_rate=specs_unmix["sample_rate"], n_fft=specs_unmix["n_fft"],
                            n_hop=specs_unmix["n_hop"], input_is_spectrogram=False)
    with open("training_specs/toy_example_naive_landmark_only_unmix.json") as f:
        specs_unmix_AI = json.load(f)
    unmix_AI = model_classes.OpenUnmixWithLandmarks(sample_rate=specs_unmix_AI["sample_rate"], landmarkCount=38)
    with open("training_specs/toy_example_naive_landmark_only_unmix_duo_objective.json") as f:
        specs_unmiM_AIO = json.load(f)
    unmiM_AIO = model_classes.OpenUnmixWithLandmarks3(sample_rate=specs_unmiM_AIO["sample_rate"], landmarkCount=38)
    with open("training_specs/toy_example_only_unmix_duo_objective_shallowmodel.json") as f:
        specs_unmiM_AO = json.load(f)
    unmiM_AO = model_classes.OpenUnmixWithLandmarks5(sample_rate=specs_unmiM_AO["sample_rate"], landmarkCount=38)
    specss = [specs_unmix, specs_unmix_AI, specs_unmiM_AO, specs_unmiM_AIO]
    models = [unmix, unmix_AI, unmiM_AO, unmiM_AIO]
    names = ["unmix", "unmix_AI", "unmiM_AO", "unmiM_AIO"]
    out = []
    out_names = []
    for i in range(0, len(specss)):
        # np.random.seed(0)
        # torch.seed(0)
        current_path = evalPaths.format(specss[i]["name"])
        currentdf: pd.DataFrame = pd.read_pickle(current_path)
        currentdf = currentdf[currentdf["target"] == "vocals"]
        currentdf = currentdf[currentdf['score'].notna()]
        arr = currentdf[currentdf["metric"]=="SAR"]["score"].array
        # print(currentdf.keys())
        print(arr.shape)
        # print(currentdf.head())
        out_names.append(names[i])
        out.append(arr)
        print(names[i], "mean\n", currentdf.groupby(["metric"]).mean())
        print(names[i], "var\n", currentdf.groupby(["metric"]).var())
        # A[2]
    plt.ylabel("SAR")
    plt.boxplot(out, labels=out_names)
    plt.show()




