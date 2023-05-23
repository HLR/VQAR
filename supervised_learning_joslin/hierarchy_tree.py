from distutils.dir_util import create_tree
import json
from os import name
import pickle
from treelib import Node, Tree
from tqdm import tqdm


class hierarchy_tree():
    def __init__(self) -> None:
        label_file = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/output_names.json"
        input_file ="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/data/dataset/scene_graph/gqa_test_formatted_scene_graph.pkl"
        name_file = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/name.json"
        self.input_data = pickle.load(open(input_file, 'rb'))
        with open(label_file) as f_l, open(name_file) as f_n:
            self.labels = json.load(f_l)
            self.names = json.load(f_n)
        self.tree = self.create_tree()

    def create_tree(self):
        outputs = {}
        all_labels = []
        for key_i, value_i in self.labels.items():
            outputs[key_i] = {}
            if key_i not in all_labels:
                all_labels.append(key_i)
            for each_v in value_i:
                if each_v not in all_labels:
                    all_labels.append(each_v)
                outputs[each_v] = {}
                outputs[each_v]['parent'] = key_i
            for key_j, value_j in self.labels.items():
                if key_i == key_j:
                    continue
                if key_i in value_j:
                    outputs[key_i]['parent'] = key_j
                    flag = 1
                    break
            else:
                outputs[key_i]['parent'] = "Thing"
    
        
        outputs["Thing"]={} 
        outputs['Thing']["parent"]= None

        ### create a tree
        dict_ = outputs
        added = set()
        tree = Tree()
        while dict_:
            for key, value in dict_.items():
                if value['parent'] in added:
                    tree.create_node(key, key, parent=value['parent'])
                    added.add(key)
                    dict_.pop(key)
                    break
                elif value['parent'] is None:
                    tree.create_node(key, key)
                    added.add(key)
                    dict_.pop(key)
                    break
        return tree






    



