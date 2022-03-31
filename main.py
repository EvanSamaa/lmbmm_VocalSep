import os
import util.dataset_util as du
import training as tr
import torch
from util.data_loader import NUSMusicTest
if __name__ == "__main__":
    # du.prepare_NUS()
    # du.analyze_timit()
    # du.prepare_timit()
    # du.prepare_NUS()
    # du.analyze_NUS()
    # du.prepare_instrumental()
    # du.prepare_NUS_landmarks()
    # A[2]
    data = NUSMusicTest("landmarks", mono=True, fixed_length=True, landmarkNoise=0.02)
    mix, groundTruth, sideinfo = data.__getitem__(0)