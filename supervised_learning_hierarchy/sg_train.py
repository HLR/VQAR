import argparse
import os
import numpy as np
import torch
import random
import json
import sys
import pickle
from sg_model import SceneGraphModel
from sg_data_loader import SceneGraphLoader
from sg_trainer import SceneGraphTrainer

# common_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
# sys.path.insert(0,common_dir)

# from utils import get_default_args


def train(args):
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    print('loading object features')
    obj_feats = np.load(args.feat_f, allow_pickle=True).item()
    feat_dim = list(obj_feats.values())[0].shape[0]
    print('feature dimension = %d' % feat_dim)

    meta_info = json.load(open(args.meta_f, 'r'))
    name1_meta = json.load(open(args.name1_path, 'r'))
    name2_meta = json.load(open(args.name2_path, 'r'))
    name3_meta = json.load(open(args.name3_path, 'r'))
    name4_meta = json.load(open(args.name4_path, 'r'))

    # read name/attribute embeddings here
    # TODO: read name and attribute embeddings here
    # name_embeddings = pickle.load(open(args.name_glove_f, 'rb'))        # numpy array of shape (n_names, glove_dim)
    # attribute_embeddings = pickle.load(open(args.attr_glove_f, 'rb'))   # numpy array of shape (n_attrs+1, glove_dim)
    # relate_embeddings = pickle.load(open(args.rel_glove_f, 'rb'))       # numpy array of shape (n_rels, glove_dim)

    # glove_dim = 300
    # name_embeddings = np.random.rand(len(meta_info['name']['idx']), glove_dim)
    # attribute_embeddings = np.random.rand(len(meta_info['attr']['idx'])+1, glove_dim)

    sg_model_dict = SceneGraphModel(
        feat_dim=feat_dim,
        n_names=meta_info['name']['num'],
        n_attrs=meta_info['attr']['num'],
        n_rels=meta_info['rel']['num'],
        # name_embeddings=name_embeddings,
        # attribute_embeddings=attribute_embeddings,
        # relate_embeddings=relate_embeddings,
        device=device
    )
    sg_model = [sg_model_dict.models["name1"], sg_model_dict.models["name2"],\
                sg_model_dict.models["name3"], sg_model_dict.models["name4"]]

    print('loading training scene graphs for %s' % args.type)
    sg_train_loader = SceneGraphLoader(
        meta_info=meta_info,
        data_f=args.train_f,
        obj_feats=obj_feats,
        type=args.type,
        batch_size=args.batch_size
    )

    print('loading validation scene graphs for %s' % args.type)
    sg_val_loader = SceneGraphLoader(
        meta_info=meta_info,
        data_f=args.val_f,
        obj_feats=obj_feats,
        type=args.type,
        batch_size=args.batch_size
    )

    sg_trainer = SceneGraphTrainer(
        model=sg_model,
        train_data_loader=sg_train_loader,
        val_data_loader=sg_val_loader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        model_dir=args.model_dir,
        type=args.type,
        meta_info_list = [meta_info, name1_meta, name2_meta, name3_meta, name4_meta]
    )

    sg_trainer.train()

if __name__ == '__main__':

    DATA_ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
    GQA_DIR = os.path.join(DATA_ROOT, 'dataset')
    GQA_SG_DIR = os.path.join(GQA_DIR, 'scene_graph')
    # DATA_ROOT = os.path.join(data_dir, "scene_graph_data")
    # DATA_ROOT = '/localscratch/bchen346/project_data/GQA'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    # parser.add_argument('--feat_dim', type=int, default=2048)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--type', default='name')   # name, relation, attribute
    parser.add_argument('--model_dir', default=DATA_ROOT+'/model_ckpts_sg_test')
    parser.add_argument('--feat_f', default=DATA_ROOT+'/features.npy')
    parser.add_argument('--bbox_f', default=GQA_DIR+'/bbox.npy')
    parser.add_argument('--name_glove_f', default=GQA_DIR+'/name_gloves.pkl')
    parser.add_argument('--attr_glove_f', default=GQA_DIR+'/attr_gloves.pkl')
    parser.add_argument('--rel_glove_f', default=GQA_DIR+'/rela_gloves.pkl')
    parser.add_argument('--transe_f', default=GQA_DIR+'/kg_emb.pkl')
    parser.add_argument('--meta_f', default=DATA_ROOT+'/gqa_info.json')
    parser.add_argument('--name1_path', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name1.json")
    parser.add_argument('--name2_path', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name2.json")
    parser.add_argument('--name3_path', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name3.json")
    parser.add_argument('--name4_path', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name4.json")
    parser.add_argument('--train_f', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/train.pickle")
    parser.add_argument('--val_f', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/val.pickle")
    parser.add_argument('--test_f', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/test.pickle")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train(args)
