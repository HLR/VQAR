import os
import json
import numpy as np
import sys
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss
import time
import torch.multiprocessing as mp
import logging
from pathlib import Path
import infer_gbi
from scipy.sparse import csr_matrix
from tensorboardX import SummaryWriter


# import concurrent.futures

supervised_learning_path = os.path.abspath(os.path.join(
    os.path.abspath(__file__), "../../supervised_learning_joslin"))
sys.path.insert(0, supervised_learning_path)

common_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.insert(0, common_path)

query_process_path = os.path.abspath(os.path.join(
    os.path.abspath(__file__), "../../query_process"))
sys.path.insert(0, query_process_path)

from query_api import TadalogInterface
from cmd_args import cmd_args
from word_idx_translator import Idx2Word
from sg_model import SceneGraphModel
from learning import get_batched_fact_probs, get_wmc
from utils import auc_score, get_recall
from debugger import debug_and_draw

log_writer = SummaryWriter(logdir=cmd_args.model_dir+"/run")

class ClauseNTrainer():

    def __init__(self,
                 train_data_loader,
                 val_data_loader,
                 test_data_loader=None,
                 model_dir=cmd_args.model_dir, n_epochs=cmd_args.n_epochs,
                 meta_f=cmd_args.meta_f, knowledge_base_dir=cmd_args.knowledge_base_dir,
                 axiom_update_size=cmd_args.axiom_update_size, reinforce=cmd_args.reinforce, replays=cmd_args.replays):

        try:
            mp.set_start_method('spawn')
        except:
            pass

        self.model_dir = model_dir

        model_exists = self._model_exists(model_dir)

        if not model_exists:
            load_model_dir = None
        else:
            load_model_dir = model_dir

        if train_data_loader is not None:
            self.train_data = train_data_loader
        if val_data_loader is not None:
            self.val_data = val_data_loader
        if test_data_loader is not None:
            self.val_data = test_data_loader

        self.is_train = test_data_loader is None
        meta_info = json.load(open(meta_f, 'r'))
        
        ### hierarchy classes
        concept_path = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/hossein_data/concepts1.json"
        kb_path = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/hossein_data/hiereachy1.json"
        with open(concept_path) as f1, open(kb_path) as f2:
            self.concepts = json.load(f1)
            self.kb = json.load(f2)
        self.CONCEPT_NUM = 500
        self.concept_dict = {}
        for id, item in enumerate(self.concepts):
            self.concept_dict[item] = id
        self.adj_matrix = self.build_matirx().cuda()

        self.idx2word = Idx2Word(meta_info)
        self.n_epochs = n_epochs
        self.dl_inter = TadalogInterface(knowledge_base_dir, meta_f)
        self.axiom_update_size = axiom_update_size
        self.reinforce = reinforce
        self.replays = replays
        self.wmc_funcs = {}

        # load dictionary from previous training results
        
        self.sg_model = SceneGraphModel(
            feat_dim=cmd_args.feat_dim,
            n_names=meta_info['name']['num'],
            n_attrs=meta_info['attr']['num'],
            n_rels=meta_info['rel']['num'],
            device=torch.device('cuda' if cmd_args.gpu >= 0 else 'cpu'),
            model_dir=load_model_dir
        )

        self.sg_model_dict = self.sg_model.models

        self.loss_func = BCELoss()

        if self.is_train:
            self.optimizers = {}
            self.schedulers = {}

            for model_type, model_info in self.sg_model_dict.items():
                if model_type == 'name1':
                    self.optimizers['name1'] = optim.Adam(
                        model_info.parameters(), lr=cmd_args.name_lr, weight_decay=1e-3)
                    self.schedulers['name1'] = StepLR(
                        self.optimizers['name1'], step_size=10, gamma=0.1)
                if model_type == 'name2':
                    self.optimizers['name2'] = optim.Adam(
                        model_info.parameters(), lr=cmd_args.name_lr, weight_decay=1e-3)
                    self.schedulers['name2'] = StepLR(
                        self.optimizers['name2'], step_size=10, gamma=0.1)
                if model_type == 'name3':
                    self.optimizers['name3'] = optim.Adam(
                        model_info.parameters(), lr=cmd_args.name_lr)
                    self.schedulers['name3'] = StepLR(
                        self.optimizers['name3'], step_size=10, gamma=0.1)
                if model_type == 'name4':
                    self.optimizers['name4'] = optim.Adam(
                        model_info.parameters(), lr=cmd_args.name_lr)
                    self.schedulers['name4'] = StepLR(
                        self.optimizers['name4'], step_size=10, gamma=0.1)
                    # self.loss_func['name'] = F.cross_entropy
                if model_type == 'relation':
                    self.optimizers['rel'] = optim.Adam(
                        model_info.parameters(), lr=cmd_args.rela_lr)
                    self.schedulers['rel'] = StepLR(
                        self.optimizers['rel'], step_size=10, gamma=0.1)
                if model_type == 'attribute':
                    self.optimizers['attr'] = optim.Adam(
                        model_info.parameters(), lr=cmd_args.attr_lr)
                    self.schedulers['attr'] = StepLR(
                        self.optimizers['attr'], step_size=10, gamma=0.1)

        #self.pool = mp.Pool(cmd_args.max_workers)
        # self.batch = cmd_args.trainer_batch
    
    def build_matirx(self):
        def adj_format(adj_mat):
        	# adj_mat is V*V tensor, which V is number of image labels.
            E = torch.count_nonzero(torch.tensor(adj_mat))
            V = adj_mat.shape[0]
            # Creating a E * V sparse matrix
            adj_mat_S = csr_matrix(adj_mat)
            EV_mat = torch.from_numpy(csr_matrix((E, V),dtype = np.int32).toarray())
            indices = adj_mat_S.nonzero()
            rows = indices[0]
            cols = indices[1]
            data = adj_mat_S.data
            for i in range(E):
                EV_mat[i][rows[i]]=data[i]
                EV_mat[i][cols[i]]=-data[i]
            return EV_mat 

        matrix = np.zeros((self.CONCEPT_NUM, self.CONCEPT_NUM))
        for i in range(len(self.concepts)):
            tmp_c = self.concepts[i]
            child = []
            for key, value in self.kb.items():
                if tmp_c in value:
                    child = value[tmp_c]
                    break
            for j in child:
                matrix[i][self.concept_dict[j]] = 1
        return adj_format(matrix)
    
  

    def _model_exists(self, model_dir):
        if model_dir is None:
            return False

        name_f1 = os.path.join(self.model_dir, "class1", 'name_best_epoch.pt')
        name_f2 = os.path.join(self.model_dir, "class2", 'name_best_epoch.pt')
        name_f3 = os.path.join(self.model_dir, "class3", 'name_best_epoch.pt')
        name_f4 = os.path.join(self.model_dir, "class4", 'name_best_epoch.pt')
        rela_f = os.path.join(self.model_dir, 'relation_best_epoch.pt')
        attr_f = os.path.join(self.model_dir, 'attribute_best_epoch.pt')

        if not os.path.exists(name_f1):
            return False
        if not os.path.exists(name_f2):
            return False
        if not os.path.exists(name_f3):
            return False
        if not os.path.exists(name_f4):
            return False
        # if not os.path.exists(rela_f):
        #     return False
        # if not os.path.exists(attr_f):
        #     return False
        return True

    def _get_optimizer(self, data_type):
        # if data_type in ['name', 'rel']:
        optimizer = self.optimizers[data_type]
        return optimizer

    def _step_all(self):
        for optim_type, optim_info in self.optimizers.items():
            optim_info.step()

    def _step_scheduler(self):
        for scheduler_type, scheduler_info in self.schedulers.items():
            scheduler_info.step()

    def _zero_all(self):
        for optim_type, optim_info in self.optimizers.items():
            optim_info.zero_grad()

    def _train_all(self):
        for model_type, model_info in self.sg_model_dict.items():
            if model_type == 'name1':
                model_info.train()
                model_info.share_memory()
            if model_type == 'name2':
                model_info.train()
                model_info.share_memory()
            if model_type == 'name3':
                model_info.train()
                model_info.share_memory()
            if model_type == 'name4':
                model_info.train()
                model_info.share_memory()
            if model_type == 'relation':
                model_info.train()
                model_info.share_memory()
            if model_type == 'attribute':
                model_info.train()
                model_info.share_memory()

    def _eval_all(self):
        for model_type, model_info in self.sg_model_dict.items():
            if model_type == 'name1':
                model_info.eval()
                model_info.share_memory()
            if model_type == 'name2':
                model_info.eval()
                model_info.share_memory()
            if model_type == 'name3':
                model_info.eval()
                model_info.share_memory()
            if model_type == 'name4':
                model_info.eval()
                model_info.share_memory()
            if model_type == 'relation':
                model_info.eval()
                model_info.share_memory()
            if model_type == 'attribute':
                model_info.eval()
                model_info.share_memory()

    def _save_all(self):
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        for model_type, model_info in self.sg_model_dict.items():
            if model_type == 'name1':
                save_f = os.path.join(self.model_dir, 'class1', 'name_best_epoch.pt')
                torch.save(model_info.state_dict(), save_f)
            if model_type == 'name2':
                save_f = os.path.join(self.model_dir, 'class2','name_best_epoch.pt')
                torch.save(model_info.state_dict(), save_f)
            if model_type == 'name3':
                save_f = os.path.join(self.model_dir, 'class3', 'name_best_epoch.pt')
                torch.save(model_info.state_dict(), save_f)
            if model_type == 'name4':
                save_f = os.path.join(self.model_dir, 'class4', 'name_best_epoch.pt')
                torch.save(model_info.state_dict(), save_f)
            if model_type == 'relation':
                save_f = os.path.join(self.model_dir, 'relation_best_epoch.pt')
                torch.save(model_info.state_dict(), save_f)
            if model_type == 'attribute':
                save_f = os.path.join(self.model_dir, 'attribute_best_epoch.pt')
                torch.save(model_info.state_dict(), save_f)

    def loss_auc_grad(self, wmc_value, wmc_idxes, probs, correct, all_oids, is_train=True):
        all_labels = [1 if obj in correct else 0 for obj in all_oids]

        if len(wmc_value) == 0:
            if self.reinforce:
                return (0, 0), -1, (0, 0)
            return 0, -1, (0, 0)

        all_labels = [1 if obj in correct else 0 for obj in all_oids]
        prob_list = [probs["name"], probs["attr"], probs["rela"]]
        prob_list = [prob for prob in prob_list if not prob == []]
        probs = torch.cat(prob_list)
        labels = []

        wmc_grads = []
        all_preds = []
        pred_ids = []
        minimal_probs = []
        for ct, oid in enumerate(all_oids):
            if oid in wmc_value:
                pred_ids.append(ct)
                all_preds.append(wmc_value[oid][0])
                if not self.reinforce:
                    minimal_probs.append(probs[wmc_idxes[oid]])
                wmc_grads.append(wmc_value[oid][1])
                labels.append(oid in correct)
            else:
                all_preds.append(0)


        all_labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
        all_pred_tensor = torch.tensor(all_preds, requires_grad=True)
        pred_tensor = torch.tensor(all_pred_tensor[pred_ids].tolist(), requires_grad=True)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        all_labels_tensor_rs = all_labels_tensor.reshape(1, -1)
        all_pred_tensor_rs = all_pred_tensor.reshape(1, -1)
        pred_tensor_rs = pred_tensor.reshape(1, -1)
        labels_tensor_rs = labels_tensor.reshape(1, -1)

        if self.reinforce:
            if labels_tensor.sum() == 0:
                if sum(all_labels) == 0:
                    return (0, 0), 1, (0, 0)
                else:
                    return (0, 0), 0, (0, 0)
            reward = (pred_tensor * labels_tensor).sum()/labels_tensor.sum()

            loss = (-probs * reward).sum()
            if torch.isnan(loss):
                print("loss is nan!")
                print(labels_tensor)
                exit()
            if loss.item() < 0:
                print("negative loss")
                print(reward.item(), -probs.sum().item(), probs, pred_tensor_rs, labels_tensor_rs)
        else:
            loss = self.loss_func(pred_tensor_rs, labels_tensor_rs)
        recall = get_recall(all_labels_tensor_rs, all_pred_tensor_rs)

        loss_grad = 0
        wmc_grad = 0

        if is_train:
            loss.backward()
            if not self.reinforce:
                grad = pred_tensor.grad
                for prob, loss_grad, wmc_grad in zip(minimal_probs, grad.tolist(), wmc_grads):
                    grad = loss_grad * wmc_grad
                    prob.backward(loss_grad * torch.tensor(wmc_grad), retain_graph=True)

        if self.reinforce:
            return (loss.item(), reward.item()), recall, (loss_grad, wmc_grad)
        return loss.item(), recall, (loss_grad, wmc_grad)

    def _batch_pass(self, batched_dp, update_axiom=True, is_train=True):

        all_labels = []
        for datapoint in batched_dp:
            # print(datapoint["question"]["clauses"])
            correct = datapoint['question']['output']
            all_labels.append([1 if obj in correct else 0 for obj in datapoint['object_ids']])


        batched_tps, batched_probs, batched_torch_probs = get_batched_fact_probs(self.sg_model, batched_dp, self.idx2word, self.dl_inter, self.concepts, self.reinforce)
        args = [(dp, tps, probs, self.dl_inter) for current_id, (dp, tps, probs) in enumerate(zip(batched_dp, batched_tps, batched_probs))]

        for probs, dp  in zip(batched_torch_probs, batched_dp):
            name_log_probs = probs['log_name']
            gbi_loss = infer_gbi.run_gbi(name_log_probs, self.sg_model.models, self.adj_matrix)
            print('yue')

        start = time.time()
        #results = [self.pool.apply_async(get_wmc, arg) for arg in args]
        results = [(get_wmc, arg) for arg in args]
        #wmcs = [res.get(timeout=2000) for res in results]
        wmcs = [res for res in results]
        end = time.time()
    
        aucs = []
        losses = []
        grads = []
        for (wmc_value, wmc_idxes), probs, dp  in zip(wmcs, batched_torch_probs, batched_dp):
            correct_oids = dp['question']['output']
            all_oids = dp['object_ids']
            loss, auc, grad = self.loss_auc_grad(wmc_value, wmc_idxes, probs, correct_oids, all_oids, is_train)
            if auc >= 0:
                aucs.append(auc)
            name_log_probs = probs['log_name']
            gbi_loss = infer_gbi.run_gbi(name_log_probs, self.sg_model.models)
            losses.append(loss+gbi_loss)
            grads.append(grad)
     
        return aucs, losses

    def _train_epoch(self, ct):

        self._train_all()
        aucs = []
        losses = []
        rewards = []

       # pbar = tqdm(self.train_data, total = self.train_data.batch_num)

        for ct, datapoint in enumerate(self.train_data):
            if self.reinforce:
                for trial in range(self.replays):
                    batch_auc, batch_loss = self._batch_pass(datapoint, True, is_train=True)
                    aucs += [auc for auc in batch_auc if auc > 0]
                    new_losses, new_rewards = zip(*batch_loss)
                    losses += new_losses
                    rewards += new_rewards
                pbar.set_description(
                    f'[loss: {np.array(losses).mean():.3f}, reward: {np.array(rewards).mean():.3f}, recall: {np.array(aucs).mean():.3f}]')
            else:
                batch_auc, batch_loss = self._batch_pass(datapoint, True, is_train=True)
                print(batch_loss)
                import sys
                sys.exit()
                # timeout, auc, loss = self._pass(datapoint, True)
                aucs += [auc for auc in batch_auc if auc >= 0]
                losses += batch_loss
                pbar.set_description(
                    f'[loss: {np.array(losses).mean()}, recall: {np.array(aucs).mean()}]')
            torch.cuda.empty_cache()

            self._step_all()
            self._zero_all()

        if self.reinforce:
            return np.mean(losses), np.mean(rewards), np.mean(aucs)
        else:
            return np.mean(losses), np.mean(aucs)

    def _val_epoch(self):
        self._eval_all()

        ct = 0
        aucs = []
        losses = []
        rewards = []

        pbar = tqdm(self.val_data, total = self.val_data.batch_num)
        with torch.no_grad():
            for datapoint in pbar:
                # if not 2379195 == datapoint[0]["question"]["image_id"]:
                #     continue
                # else:
                #     datapoint[0]['question']['clauses'] = \
                #     [{'clause_id': '2379195_32_0', 'function': 'Initial', 'input_clause_id': None},
                #         {'clause_id': '2379195_32_1', 'function': 'Relate', 'input_clause_id': '2379195_32_0', 'text_input': 'left'},
                #         {'clause_id': '2379195_32_2', 'function': 'Find_Attr', 'input_clause_id': '2379195_32_1', 'text_input': "tall"},
                #         {'clause_id': '2379195_32_3', 'function': 'Hypernym_Find', 'input_clause_id': '2379195_32_2', 'text_input': "animal"}]
                #     datapoint[0]['question']['output'] = [554396]

                batch_auc, batch_loss = self._batch_pass(datapoint, True, is_train=False)
                aucs += [auc for auc in batch_auc if auc >= 0]

                if self.reinforce:
                    new_losses, new_rewards = zip(*batch_loss)
                    losses += new_losses
                    rewards += new_rewards
                    pbar.set_description(
                        f'[loss: {np.array(losses).mean():.3f}, reward: {np.array(rewards).mean():.3f}, recall: {np.array(aucs).mean():.3f}]')
                else:
                    losses += batch_loss
                    pbar.set_description(
                        f'[loss: {np.array(losses).mean()}, recall: {np.array(aucs).mean()}]')

        if self.reinforce:
            return np.mean(losses), np.mean(rewards), np.mean(aucs)
        else:
            return np.mean(losses), np.mean(aucs)

    def train(self):
        assert self.is_train
        best_val_loss = np.inf

        # val_loss, val_acc = self._val_epoch()
        for epoch in range(self.n_epochs):
            if self.reinforce:
                train_loss, train_reward, train_acc = self._train_epoch(epoch)
                val_loss, val_reward, val_acc = self._val_epoch()
                print(
                    '[Epoch %d/%d] [training loss: %.2f, reward: %.2f, recall: %.2f] [validation loss: %.2f, reward: %.2f, recall: %.2f]' %
                    (epoch, self.n_epochs, train_loss, train_reward, train_acc, val_loss, val_reward, val_acc)
                )
            else:
                train_loss, train_acc = self._train_epoch(epoch)
                val_loss, val_acc = self._val_epoch()

                log_writer.add_scalar('Loss/train', float(train_loss), epoch)
                log_writer.add_scalar('Loss/val', float(val_loss), epoch)
                log_writer.add_scalar('Acc/train', float(train_acc), epoch)
                log_writer.add_scalar('Acc/val', float(val_acc), epoch)

                print(
                    '[Epoch %d/%d] [training loss: %.2f, recall: %.2f] [validation loss: %.2f, recall: %.2f]' %
                    (epoch, self.n_epochs, train_loss, train_acc, val_loss, val_acc)
                )

                logging.info(
                '[Epoch %d/%d] [training loss: %.2f, recall: %.2f] [validation loss: %.2f, recall: %.2f]' %
                (epoch, self.n_epochs, train_loss, train_acc, val_loss, val_acc)
                )

            self._step_scheduler()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print('saving best models')
                self._save_all()

    def test(self):
        assert not self.is_train
        if self.reinforce:
            test_loss, test_reward, test_acc = self._val_epoch()
            print('[test loss: %.2f, reward: %.2f, recall: %.2f]' % (test_loss, test_reward, test_acc))
        else:
            test_loss, test_acc = self._val_epoch()
            print('[test loss: %.2f, recall: %.2f]' % (test_loss, test_acc))
