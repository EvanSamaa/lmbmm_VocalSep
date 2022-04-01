import os
import util.dataset_util as du
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
    du.prepare_NUS_landmarks()
    A[2]
    data = NUSMusicTest("landmarks", mono=True, fixed_length=True, landmarkNoise=0.02)
    mix, groundTruth, sideinfo = data.__getitem__(0)
    print(sideinfo.shape)

