import argparse
import os
import numpy as np
import torch
import random
import json
from sg_model import SceneGraphModel
from sg_data_loader import SceneGraphLoader
from sg_trainer import SceneGraphTrainer


def test(args):
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

    sg_model_dict = SceneGraphModel(
        feat_dim=feat_dim,
        n_names=meta_info['name']['num'],
        n_attrs=meta_info['attr']['num'],
        n_rels=meta_info['rel']['num'],
        device=device,
        model_dir=args.model_dir
    )
    # types = ['name', 'relation', 'attribute']
    # types.extend(['attr:%d' % idx for idx in range(50)])
    sg_model = [sg_model_dict.models["name1"].eval(), sg_model_dict.models["name2"].eval(),\
                sg_model_dict.models["name3"].eval(), sg_model_dict.models["name4"].eval()]

    # sg_model = sg_model_dict.models[type]
    #sg_model.eval()
    print('loading testing scene graphs for %s' % type)
    sg_test_loader = SceneGraphLoader(
        meta_info=meta_info,
        data_f=args.test_f,
        obj_feats=obj_feats,
        type=args.type,
        batch_size=args.batch_size,
        drop_last=False
    )
        

    sg_trainer = SceneGraphTrainer(
        model=sg_model,
        train_data_loader=None,
        val_data_loader=None,
        test_data_loader=sg_test_loader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        model_dir=args.model_dir,
        type=args.type,
        topk=args.topk,
        meta_info_list = [meta_info, name1_meta, name2_meta, name3_meta, name4_meta]
    )

    sg_trainer.test()

if __name__ == '__main__':
    # DATA_ROOT = os.getenv('HOME') + '/project_data/CRIC'
    DATA_ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
    GQA_DIR = os.path.join(DATA_ROOT, 'dataset')
    GQA_SG_DIR = os.path.join(GQA_DIR, 'scene_graph')
    # DATA_ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
    # DATA_ROOT = os.path.join(data_dir, "scene_graph_data")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--type', default='name')
    parser.add_argument('--model_dir', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/data/model_ckpts_sg_test/")
    parser.add_argument('--feat_f', default=DATA_ROOT+'/features.npy')
    parser.add_argument('--train_f', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/train.pickle")
    parser.add_argument('--val_f', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/val.pickle")
    parser.add_argument('--test_f', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/val.pickle")
    parser.add_argument('--name1_path', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name1.json")
    parser.add_argument('--name2_path', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name2.json")
    parser.add_argument('--name3_path', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name3.json")
    parser.add_argument('--name4_path', default="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name4.json")
    parser.add_argument('--meta_f', default=DATA_ROOT+'/gqa_info.json')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    test(args)
