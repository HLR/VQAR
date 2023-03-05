import torch
from torch.distributions import Categorical

import os
import sys
import multiprocessing as mp
import gc
import time


common_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.insert(0, common_path)

from cmd_args import cmd_args
import numpy as np

# obtain all atttributes mentioned in the question
def get_attrs(question):
    program = question['clauses']
    attrs = []
    for clause in program:
        if clause['function'] == "Find_Attr" :
            attr = clause['text_input']
            if not type(attr) == type(None):
                attrs.append(attr)
    return attrs

def get_batched_fact_probs(sg_models, datapoints, idx2word, dl_inter, hierchy_labels=None,
                     reinforce=cmd_args.reinforce):


    batched_obj_features = []
    batched_relation_features = []

    batched_attr_idxes = []
    batched_rela_idxes = []
    batched_name_tps = []
    batched_attr_tps = []
    batched_rela_tps = []

    split_obj_ids = []
    split_rela_ids = []
    #name_choices = idx2word.get_names()
    name_choices = hierchy_labels
    for datapoint in datapoints:
        object_ids = datapoint['object_ids']
        object_features = datapoint['object_feature']
        bboxes = datapoint['scene_graph']['bboxes']
        query = datapoint['question']['clauses']
        object_feature_dict = {str(object_id): feature for object_id, feature in zip(object_ids, object_features)}
        batched_obj_features += object_features

        # generate object attributes
        candidate_attrs = get_attrs(datapoint['question'])
        candidate_attr_idxes = [idx2word.attr_to_idx(candidate_attr) for candidate_attr in candidate_attrs]
        candidate_attr_idxes = [ x for x in candidate_attr_idxes if x is not None]
        batched_attr_idxes.append(candidate_attr_idxes)
        current_attr_tps = []
        current_name_tps = []

        for object_id in object_ids:
            for candidate_attr in candidate_attrs:
                current_attr_tps.append((object_id, candidate_attr))
            for name in name_choices:
                current_name_tps.append((object_id, name))

        batched_attr_tps.append(current_attr_tps)
        batched_name_tps.append(current_name_tps)

        candidate_relations = dl_inter.get_relas(query)
        candidate_rela_idxes = [idx2word.rela_to_idx(candidate_rela) for candidate_rela in candidate_relations]
        candidate_rela_idxes = [x for x in candidate_rela_idxes if x is not None]
        batched_rela_idxes.append(candidate_rela_idxes)
        current_rela_tps = []

        for sub in object_ids:
            for obj in object_ids:
                if sub == obj:
                    continue

                sub_feat_np_array = object_feature_dict[str(sub)]
                obj_feat_np_array = object_feature_dict[str(obj)]
                sub_bbox_np_array = np.array(bboxes[int(sub)])
                obj_bbox_np_array = np.array(bboxes[int(obj)])
                batched_relation_features.append(np.concatenate([sub_feat_np_array, obj_feat_np_array, sub_bbox_np_array, obj_bbox_np_array]))
                for rela in candidate_relations:
                    current_rela_tps.append((rela, sub, obj))

        batched_rela_tps.append(current_rela_tps)
        split_obj_ids.append(len(batched_obj_features))
        split_rela_ids.append(len(batched_relation_features))

    # Generate object names

    # RL mod
    # Sample discrete graphs from here and return
    # will have 0, 1 for the probability
    logits1, name_probs1 = sg_models.batch_predict(
            type='name1',
            inputs=batched_obj_features,
            batch_split=split_obj_ids
        )
    
    logits2, name_probs2 = sg_models.batch_predict(
            type='name2',
            inputs=batched_obj_features,
            batch_split=split_obj_ids
        )
    logits3, name_probs3 = sg_models.batch_predict(
            type='name3',
            inputs=batched_obj_features,
            batch_split=split_obj_ids
        )
    logits4, name_probs4 = sg_models.batch_predict(
            type='name4',
            inputs=batched_obj_features,
            batch_split=split_obj_ids
        )

    _, attr_probs = sg_models.batch_predict(
            type='attribute',
            inputs=batched_obj_features,
            batch_split=split_obj_ids
        )

    if not len(batched_relation_features) == 0:
        _, rel_probs = sg_models.batch_predict(
                    type='relation',
                    inputs=batched_relation_features,
                    batch_split=split_rela_ids
                )

    if reinforce:
        # sample in batch
        name_dist = Categorical(name_probs)
        name_samp = name_dist.sample()
        # ideally sample multiple attributes, but with more samples this should be OK
        attr_dist = Categorical(attr_probs)
        attr_samp = attr_dist.sample()
        if not len(batched_relation_features) == 0:
            rel_dist = Categorical(rel_probs)
            rel_samp = rel_dist.sample()

    current_obj_id = 0
    current_rela_id = 0

    batched_tps = []
    batched_probs = []
    batched_torch_probs = []

    for name_tps, attr_tps, rela_tps, rela_idxes, attr_idxes, next_obj_id, next_rela_id in zip(batched_name_tps,
        batched_attr_tps, batched_rela_tps, batched_rela_idxes, batched_attr_idxes, split_obj_ids, split_rela_ids):

        tps  = {}
        torch_probs = {}
        probs = {}
        tps['name'] = name_tps
        if reinforce:
            torch_probs['name'] = name_dist.log_prob(name_samp)[current_obj_id: next_obj_id]
            probs["name"] = torch.zeros(len(name_tps))  # may need some epsilon term for wmc calculation
            # since we flatten we need to add the sample indicies to the flattened zero vector
            offset = torch.arange(len(name_samp)) * len(name_choices)
            probs["name"][name_samp + offset] = 1
            probs["name"] = probs["name"].tolist()
        else:
            name_probs = torch.cat([name_probs1, name_probs2[:,:-1], name_probs3[:,:-1], name_probs4[:,:-1]],dim=-1)
            name_log_probs = torch.cat([torch.log(name_probs1), torch.log(name_probs2)[:,:-1], torch.log(name_probs3)[:,:-1],
                                   torch.log(name_probs4)[:,:-1]], dim=-1)
            torch_probs['log_name'] = name_log_probs[current_obj_id: next_obj_id]
            torch_probs['name'] = name_probs[current_obj_id: next_obj_id].reshape(-1)
            probs["name"] = torch_probs['name'].tolist()
            # print(torch_probs['name'], torch_probs['name'].shape)

        tps['attr'] = attr_tps
        if len(attr_tps) > 0:
            if reinforce:
                # print('ATTR_TP')
                # get probs of attr indicies, or use all probs?
                # idx_probs = torch.stack([attr_dist.log_prob(torch.tensor(idx)) for idx in attr_idxes], dim=1)
                # torch_probs['attr'] = idx_probs.flatten()
                # all probs
                torch_probs['attr'] = attr_dist.log_prob(attr_samp)[current_obj_id: next_obj_id]
                probs['attr'] = []
                for sample in attr_samp:
                    for idx in attr_idxes:
                        probs['attr'].append(1.0 if sample == idx else 0.0)
            else:
                torch_probs['attr'] = attr_probs[current_obj_id: next_obj_id][:, attr_idxes].reshape(-1)
                probs['attr'] = torch_probs['attr'].tolist()
        else:
            torch_probs['attr'] = []
            probs['attr'] = []

        tps['rela'] = rela_tps
        if len(rela_tps) > 0:
            if reinforce:
                # print('RELA_TP')
                # idx_probs = torch.stack([rel_dist.log_prob(torch.tensor(idx)) for idx in rela_idxes], dim=1)
                # torch_probs['rela'] = idx_probs.flatten()
                torch_probs['rela'] = rel_dist.log_prob(rel_samp)[current_obj_id: next_obj_id]
                probs['rela'] = torch_probs['rela'].tolist()
                probs['rela'] = []
                for sample in rel_samp:
                    for idx in rela_idxes:
                        probs['rela'].append(1.0 if sample == idx else 0.0)
            else:
                torch_probs['rela'] = rel_probs[current_rela_id: next_rela_id][:, rela_idxes].reshape(-1)
                probs['rela'] = torch_probs['rela'].tolist()
        else:
            torch_probs['rela'] = []
            probs['rela'] = []

        current_rela_id = next_rela_id
        current_obj_id = next_obj_id
        batched_tps.append(tps)
        batched_probs.append(probs)
        batched_torch_probs.append(torch_probs)

    return batched_tps, batched_probs, batched_torch_probs


def get_wmc(datapoint, tps, probs, dl_inter, wmc_func=None):

    query = datapoint['question']['clauses']
    all_obj_ids = datapoint['question']['input']

    # scene_prog_tp = dl_inter.fact_prob_to_tp(facts)
    if wmc_func is None:
        wmc_func, raw_weights = dl_inter.get_rust_wmc_funcs(tps, probs, query, all_obj_ids)
    wmc_value = dl_inter.get_wmc(wmc_func, raw_weights)

    return wmc_value
