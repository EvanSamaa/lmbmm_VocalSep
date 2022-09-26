import json
import math
landmark_data = "C:\Users\evansamaa\Desktop\lmbmm_VocalSep\jaliVertex2Landmark_dict.json"
with open(landmark_data) as f:
    V2L = json.load(f)
landmark_sets = ["nose", "lips", "lower_face"]
sorted_vertex_2_landmark_list = []

for lm_set in landmark_sets:
    marks = V2L[lm_set]
    for mark in marks.keys():
        cmds.select("head_lorez.vtx[{}]".format(mark), add=True)
        sorted_vertex_2_landmark_list.append([mark, V2L[lm_set][mark]])
sorted_vertex_2_landmark_list.sort(key=lambda x: x[1])

input_file_template = "F:/MASC/lmbmm_vocal_sep_data/NUS/test/{}.json"
out_file_template = "F:/MASC/lmbmm_vocal_sep_data/NUS/test_landmarks_raw/{}_raw.json"
for i in range(0, 0):
    # prep output file format
    out_dict = {"fps": 24, "landmarks": {}}
    for item in sorted_vertex_2_landmark_list:
        out_dict["landmarks"][item[1]] = []
    input_file_path = input_file_template.format(i)
    with open(input_file_path) as f:
        data = json.load(f)
    data = json.loads(data)
    out_dict["t_max"] = data["t_max"]
    out_dict["t_min"] = data["t_min"]
    # generate curves
    gen_mvp_curves(input_file_path)
    # obtain the raw vertex positions/landmarks
    # the time frame will be from 0 -> ceil(t_max * 24)
    time_frame = math.ceil((data["t_max"]-data["t_min"]) * 24)
    for t in range(0, int(time_frame)):
        cmds.currentTime(t+data["t_min"]*24, edit=True)
        for i_vertex, i_lm in sorted_vertex_2_landmark_list:
            pos = cmds.pointPosition("head_lorez.vtx[{}]".format(i_vertex))
            out_dict["landmarks"][i_lm].append(pos)

    out_file_path = out_file_template.format(i)
    with open(out_file_path, "w") as outfile:
        json.dump(out_dict, outfile)

input_file_template = "F:/MASC/lmbmm_vocal_sep_data/NUS/train/{}.json"
out_file_template = "F:/MASC/lmbmm_vocal_sep_data/NUS/train_landmarks_raw/{}_raw.json"
for i in range(630, 854):
    # prep output file format
    out_dict = {"fps": 24, "landmarks": {}}
    for item in sorted_vertex_2_landmark_list:
        out_dict["landmarks"][item[1]] = []
    input_file_path = input_file_template.format(i)
    with open(input_file_path) as f:
        data = json.load(f)
    data = json.loads(data)
    out_dict["t_max"] = data["t_max"]
    out_dict["t_min"] = data["t_min"]
    # generate curves
    gen_mvp_curves(input_file_path)
    # obtain the raw vertex positions/landmarks
    # the time frame will be from 0 -> ceil(t_max * 24)
    time_frame = math.ceil((data["t_max"]-data["t_min"]) * 24)
    for t in range(0, int(time_frame)):
        cmds.currentTime(t+data["t_min"]*24, edit=True)
        for i_vertex, i_lm in sorted_vertex_2_landmark_list:
            pos = cmds.pointPosition("head_lorez.vtx[{}]".format(i_vertex))
            out_dict["landmarks"][i_lm].append(pos)

    out_file_path = out_file_template.format(i)
    with open(out_file_path, "w") as outfile:
        json.dump(out_dict, outfile)