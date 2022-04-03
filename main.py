import os
# import util.dataset_util as du
from util.dataset_util import prepare_NUS_landmarks
import training as tr
import util.jali_curve_generation as jg
import torch
from util.data_loader import NUSMusicTest
if __name__ == "__main__":
    # du.prepare_NUS()
    # du.analyze_timit()
    # du.prepare_timit()
    # du.prepare_NUS()
    # du.analyze_NUS()
    # du.prepare_instrumental()

    # data_set_root = "F:/MASC/lmbmm_vocal_sep_data/NUS/train"
    # path_to_text_sequences = "F:/MASC/lmbmm_vocal_sep_data/NUS/train_landmarks_raw"
    # from matplotlib import pyplot as plt
    # import numpy as np
    # arr = []
    # for i in range(0, 854):
    #     audio = torch.load(os.path.join(data_set_root, '{}.pt'.format(i)))
    #     landmarks = torch.load(os.path.join(path_to_text_sequences, '{}_processed.pt'.format(i)))[:, :, 0:2]
    #     arr.append(abs(audio.shape[0]/16000-landmarks.shape[0]/24))
    # plt.plot(np.array(arr))
    # plt.show()
    # i = 0
    # for i in range(170, 270):
    #     ja = jg.JaliVoCa_animation("E:/MASC/lmbmm_vocal_sep_data/NUS/test/{}.pt".format(i), "E:/MASC/lmbmm_vocal_sep_data/NUS/test/{}.TextGrid".format(i),
    #                                "E:/MASC/lmbmm_vocal_sep_data/NUS/test_landmarks/{}.json".format(i))
    #     ja.generate_curves()
    #     print("{}.pt is completed".format(i))
    # for i in range(181, 854):
    #     ja = jg.JaliVoCa_animation("E:/MASC/lmbmm_vocal_sep_data/NUS/train/{}.pt".format(i), "E:/MASC/lmbmm_vocal_sep_data/NUS/train/{}.TextGrid".format(i),
    #                                "E:/MASC/lmbmm_vocal_sep_data/NUS/train_landmarks/{}.json".format(i))
    #     ja.generate_curves()
    #     print("{}.pt is completed".format(i))
    prepare_NUS_landmarks()

    # data = NUSMusicTest("landmarks", mono=True, fixed_length=True, landmarkNoise=0.02)
    # mix, groundTruth, sideinfo = data.__getitem__(0)
    # print(sideinfo.shape)

