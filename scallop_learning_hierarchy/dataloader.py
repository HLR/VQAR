import pickle
import json
import sys
import numpy as np
import os

common_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.insert(0, common_path)
from utils import to_image_id_dict
from cmd_args import cmd_args

class DataLoader():

    def __init__(self, scene_graphs, image_data, questions, features, n=100):

        self.scene_graphs = scene_graphs
        self.image_data = image_data
        self.questions = questions
        self.features = features
        self.image_data = to_image_id_dict(image_data)
        all_questions = min(len(questions), n)
        self.batch_num = int(np.floor(all_questions / cmd_args.batch_size))
        self.current_batch = 0

    def __next__(self):

        batched_question = self.questions[self.current_batch * cmd_args.batch_size: (self.current_batch+1) * cmd_args.batch_size]
        self.current_batch += 1
        tasks = []

        for question in batched_question:

            if q_length is not None:
                current_len = len(question['clauses'])
                if not current_len == q_length:
                    continue

            # functions = question["clauses"]
            image_id = question["image_id"]
            scene_graph = scene_graphs[image_id]
            cur_image_data = image_data[image_id]

            if not scene_graph['image_id'] == image_id or not cur_image_data['image_id'] == image_id:
                raise Exception("Mismatched image id")

            info = {}
            info['image_id'] = image_id
            info['scene_graph'] = scene_graph
            info['url'] = cur_image_data['url']
            info['question'] = question
            info['object_feature'] = []
            info['object_ids'] = []

            for obj_id in scene_graph['names'].keys():
                info['object_ids'].append(obj_id)
                info['object_feature'].append(features.get(obj_id))
            tasks.append(info)
        return tasks

    def __iter__(self):
        self.current_batch = 0
        return self

class FileDataLoader(DataLoader):

    def __init__(self, formatted_scene_graph_path, vg_image_data_path, questions_path, features_path, n=100):
        with open(questions_path, 'rb') as questions_file:
            questions = pickle.load(questions_file)

        with open(formatted_scene_graph_path, 'rb') as vg_scene_graphs_file:
            scene_graphs = pickle.load(vg_scene_graphs_file)

        with open(vg_image_data_path, 'r') as vg_image_data_file:
            image_data = json.load(vg_image_data_file)

        features = np.load(features_path, allow_pickle=True)
        features = features.item()

        image_data = to_image_id_dict(image_data)
        super().__init__(scene_graphs, image_data, questions, features, n)

class ListDataLoader():

    def __init__(self, task_list):
        self.task_list = task_list
        self.batch_num = int(np.floor(len(task_list) / cmd_args.batch_size))
        self.current_batch = 0

    def __next__(self):
        if self.batch_num == self.current_batch:
            raise StopIteration
        batched_tasks = self.task_list[self.current_batch * cmd_args.batch_size: (self.current_batch+1) * cmd_args.batch_size]
        self.current_batch += 1
        return batched_tasks

    def __iter__(self):
        self.current_batch = 0
        return self