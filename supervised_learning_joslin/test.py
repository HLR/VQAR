import numpy as np
import pickle

path = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/train.pickle"
data = pickle.load(open(path, 'rb'))
count = 0
for item in data:
    for key,value in item['names'].items():
        if value == 200:
            count += 1
    # for key, lables in item['hierachy_names'].items():
    #     if lables[-1] == 0:
    #         count += 1
print('yue')
# file1 = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/model_res/val_res.npy"

# data1 = np.load(file1, allow_pickle=True).item()
# print('yue')