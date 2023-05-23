from curses.panel import bottom_panel
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
import os
import sys
from random import sample
from hierarchy_tree import hierarchy_tree

common_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.insert(0, common_path)

from utils import auc_score, to_binary_labels

class SceneGraphTrainer:
    def __init__(self, model, train_data_loader, val_data_loader, n_epochs, lr, device, model_dir, type, test_data_loader=None, topk=5, meta_info_list=None):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.is_train = True
        if test_data_loader is not None:
            self.val_data_loader = test_data_loader
            self.is_train = False
        self.n_epochs = n_epochs
        self.device = device
        self.model1, self.model2, self.model3, self.model4 = model[0].to(device),model[1].to(device), model[2].to(device), model[3].to(device)
        self.model_dir = model_dir
        self.type = type
        self.topk = topk
        self.meta_info = meta_info_list
        self.meta_info_verse = self.meta_info.copy()
        self.meta_info_verse[1] = {value:key for key, value in self.meta_info_verse[1].items()}
        self.meta_info_verse[2] = {value:key for key, value in self.meta_info_verse[2].items()}
        self.meta_info_verse[3] = {value:key for key, value in self.meta_info_verse[3].items()}
        self.meta_info_verse[4] = {value:key for key, value in self.meta_info_verse[4].items()}

        self.tree = hierarchy_tree().tree

        if self.is_train:
            self.optim1 = optim.Adam(self.model1.parameters(), lr=lr)
            self.optim2 = optim.Adam(self.model2.parameters(), lr=lr)
            self.optim3 = optim.Adam(self.model3.parameters(), lr=lr)
            self.optim4 = optim.Adam(self.model4.parameters(), lr=lr)
            self.scheduler1 = StepLR(self.optim1, step_size=15, gamma=0.3)
            self.scheduler2 = StepLR(self.optim2, step_size=15, gamma=0.3)
            self.scheduler3 = StepLR(self.optim3, step_size=15, gamma=0.3)
            self.scheduler4 = StepLR(self.optim4, step_size=15, gamma=0.3)
        self.loss = CrossEntropyLoss()
        if type == 'attribute':
            self.loss = BCEWithLogitsLoss()

    def _pass(self, data, train=True):
        data = [d.to(self.device) for d in data]

        feats, labels, labels1, labels2, labels3, labels4 = data

        if train:
            l3_true = torch.where(labels3 != len(self.meta_info[3]))[0]
            l3_false = torch.where(labels3 == len(self.meta_info[3]))[0]
            l3_sample = sample(l3_false.tolist(), l3_true.shape[0])
            l3_index = l3_true.tolist() + l3_sample
            l3_index = torch.tensor(l3_index).cuda()
            labels3 = labels3[l3_index]
            feats3 = feats[l3_index]

            l4_true = torch.where(labels4 != len(self.meta_info[4]))[0]
            l4_false = torch.where(labels4 == len(self.meta_info[4]))[0]
            l4_sample = sample(l4_false.tolist(), l4_true.shape[0])
            l4_index = l4_true.tolist() + l4_sample
            if not l4_index:
                l4_index= [i for i in range(10)]
            l4_index = torch.tensor(l4_index).cuda()
            labels4 = labels4[l4_index]
            feats4 = feats[l4_index]

            logits1 = self.model1(feats)
            logits2 = self.model2(feats)
            logits3 = self.model3(feats3)
            logits4 = self.model4(feats4)
        else:
            logits1 = self.model1(feats)
            logits2 = self.model2(feats)
            logits3 = self.model3(feats)
            logits4 = self.model4(feats)
        if not len(labels1.shape) == 1:
            labels1 = labels1.float()
        if not len(labels2.shape) == 1:
            labels2 = labels2.float()
        if not len(labels3.shape) == 1:
            labels3 = labels3.float()
        if not len(labels4.shape) == 1:
            labels4 = labels4.float()
        loss1 = self.loss(logits1, labels1)
        loss2 = self.loss(logits2, labels2)
        loss3 = self.loss(logits3, labels3)
        loss4 = self.loss(logits4, labels4)

        #binary_labels = to_binary_labels(labels, logits.shape[-1])
        # accuracy = auc_score(binary_labels, logits)
        '''
        if self.type != 'attribute':
            if logits1.shape[-1] < self.topk:
                self.topk = logits1.shape[-1]

            _, pred = logits.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            correct_k = correct[:self.topk].reshape(-1).float().sum(0, keepdim=True)
            accuracy = correct_k.mul_(100.0 / logits.shape[0]).item()

        else:
            if logits.shape[-1] < self.topk:
                self.topk = logits.shape[-1]

            _, pred = logits.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = torch.sum(labels.gather(1, pred.t()), dim=1)
            correct_label = torch.clamp(torch.sum(labels, dim = 1), 0, self.topk)
            accuracy = torch.mean(correct / correct_label).item()
        '''
        _, predicted1 = torch.max(logits1, 1)
        _, predicted2 = torch.max(logits2, 1)
        _, predicted3 = torch.max(logits3, 1)
        _, predicted4 = torch.max(logits4, 1)
        # if self.type == 'attribute':
        #     labels = labels.max(dim=-1)[1]



        #### check the accuracy between the original label and the predicted label
        """
        new_predict1 = []
        new_predict2 = []
        new_predict3 = []
        new_predict4 = []

        total_index = self.meta_info[0]['name']['idx']
        for i1 in predicted1:
            i1 = str(i1.cpu().tolist())
            tmp_label1 = total_index[self.meta_info[1][i1]]
            new_predict1.append(tmp_label1)
        for i2 in predicted2:
            try:
                i2 = str(i2.cpu().tolist())
                tmp_label2 = total_index[self.meta_info[2][i2]]
            except KeyError:
                tmp_label2 = -1
            new_predict2.append(tmp_label2)
        for id3, i3 in enumerate(predicted3):
            if predicted2[id3] == -1:
                tmp_label3 = -1
            else:
                try:
                    i3 = str(i3.cpu().tolist())
                    tmp_label3 = total_index[self.meta_info[3][i3]]
                except KeyError:
                    tmp_label3 = -1
            new_predict3.append(tmp_label3)
        for id4, i4 in enumerate(predicted4):
            if predicted3[id4] == -1:
                tmp_label4 = -1
            else:
                try:
                    i4 = str(i4.cpu().tolist())
                    tmp_label4 = total_index[self.meta_info[4][i4]]
                except KeyError:
                    tmp_label4 = -1
            new_predict4.append(tmp_label4)
        new_resulsts = list(map(list, zip(*[new_predict1, new_predict2, new_predict3, new_predict4])))
        correct = 0
        for id, item in enumerate(new_resulsts):
            if labels[id] in item:
                correct += 1
        """
     
        
        # hierarchy flat evaluation
        # total_list = []
        # for each_example in range(predicted1.shape[0]):
        #     correct_count = 0  
        #     if predicted1[each_example] == labels1[each_example]:
        #         correct_count += 1
        #     if predicted2[each_example] == labels2[each_example]:
        #         correct_count += 1
        #     if predicted3[each_example] == labels3[each_example]:
        #         correct_count += 1
        #     if predicted4[each_example] == labels4[each_example]:
        #         correct_count += 1
        #     total_list.append(correct_count/4)
        
        # hierarchy evaluation
        total_list = []
        for each_example in range(predicted1.shape[0]):
            
            pred1 = predicted1[each_example]
            pred2 = predicted2[each_example]
            pred3 = predicted3[each_example]
            pred4 = predicted4[each_example]
    
            if pred2 == len(self.meta_info[2]):
                pred_name = self.meta_info[1][str(pred1.detach().item())]
            elif pred3 == len(self.meta_info[3]):
                pred_name = self.meta_info[2][str(pred2.detach().item())]
            elif pred4 == len(self.meta_info[4]):
                pred_name = self.meta_info[3][str(pred3.detach().item())]
            else:
                pred_name = self.meta_info[4][str(pred4.detach().item())]
      
            if pred_name in self.meta_info_verse[1].keys():
                label1 = int(self.meta_info_verse[1][pred_name])
                label2 = len(self.meta_info[2])
                label3 = len(self.meta_info[3])
                label4 = len(self.meta_info[4])
                flag = 1
            elif pred_name in self.meta_info_verse[2].keys():
                pred_name1 = self.tree.get_node(pred_name).bpointer
                label1 =  int(self.meta_info_verse[1][pred_name1])
                label2 =  int(self.meta_info_verse[2][pred_name])
                label3 = len(self.meta_info[3])
                label4 = len(self.meta_info[4])
                flag = 2
            elif pred_name in self.meta_info_verse[3].keys():
                pred_name2 = self.tree.get_node(pred_name).bpointer
                pred_name1 = self.tree.get_node(pred_name2).bpointer
                label1 =  int(self.meta_info_verse[1][pred_name1])
                label2 =  int(self.meta_info_verse[2][pred_name2])
                label3 = int(self.meta_info_verse[3][pred_name])
                label4 = len(self.meta_info[4])
                flag = 3
            elif pred_name in self.meta_info_verse[4].keys():
                pred_name3 = self.tree.get_node(pred_name).bpointer
                pred_name2 = self.tree.get_node(pred_name3).bpointer
                pred_name1 = self.tree.get_node(pred_name2).bpointer
                label1 =  int(self.meta_info_verse[1][pred_name1])
                label2 =  int(self.meta_info_verse[2][pred_name2])
                label3 = int(self.meta_info_verse[3][pred_name3])
                label4 = int(self.meta_info_verse[4][pred_name])
                flag = 4
            
            
            correct_count = 0
            if label1 == labels1[each_example]:
                correct_count += 1
            if label2 == labels2[each_example]:
                correct_count += 1
            if label3  == labels3[each_example]:
                correct_count += 1
            if label4 == labels4[each_example]:
                correct_count += 1
            total_list.append(correct_count/4)

            
        '''
        # variable evaluation
        total_list = []
        bottom_list = []
        for each_example in range(predicted1.shape[0]):
            label1 = predicted1[each_example]
            label2 = predicted2[each_example]
            label3 = predicted3[each_example]
            label4 = predicted4[each_example]

            if labels2[each_example] == len(self.meta_info[2]):
                flag = 1
            elif labels3[each_example] == len(self.meta_info[3]):
                flag = 2
            elif labels4[each_example] == len(self.meta_info[4]):
                flag = 3
            else:
                flag = 4

            if flag == 1:
                bottom_list.append(1)
                if label1 == str(labels1[each_example].item()):
                    total_list.append(1)
                else:
                    total_list.append(0)
            elif flag == 2:
                bottom_list.append(2)
                if label1 == labels1[each_example]:
                    if label2 == labels2[each_example]:
                        total_list.append(2)
                    else:
                        total_list.append(1)
                else:
                    total_list.append(0)
            elif flag == 3:
                bottom_list.append(3)
                if label1 == labels1[each_example]:
                    if label2 == labels2[each_example]:
                        if label3 == labels3[each_example]:
                            total_list.append(3)
                        else:
                            total_list.append(2)
                    else:
                        total_list.append(1)
                else:
                    total_list.append(0)
            elif flag == 4:
                bottom_list.append(4)
                if label1 == labels1[each_example]:
                    if label2 == labels2[each_example]:
                        if label3 == labels3[each_example]:
                            if label4 == labels4[each_example]:
                                total_list.append(4)
                            else:
                                total_list.append(3)
                        else:
                            total_list.append(2)
                    else:
                        total_list.append(1)
                else:
                    total_list.append(0)

        '''
        '''
        # down-sampling
        # predicted2 = predicted2[torch.where(labels2 == len(self.meta_info[2]))[0]]
        # labels2 = labels2[torch.where(labels2 == len(self.meta_info[2]))[0]]

        # predicted3 = predicted3[torch.where(labels3 == len(self.meta_info[3]))[0]]
        # labels3 = labels3[torch.where(labels3 == len(self.meta_info[3]))[0]]

        # predicted4 = predicted4[torch.where(labels4 == len(self.meta_info[4]))[0]]
        # labels4 = labels4[torch.where(labels4 == len(self.meta_info[4]))[0]]


        # l2_true = torch.where(labels2 != len(self.meta_info[2]))[0]
        # predicted2 = predicted2[l2_true]
        # labels2 = labels2[l2_true]

        # l3_true = torch.where(labels3 != len(self.meta_info[3]))[0]
        # predicted3 = predicted3[l3_true]
        # labels3 = labels3[l3_true]

        # l4_true = torch.where(labels4 != len(self.meta_info[4]))[0]
        # predicted4 = predicted4[l4_true]
        # labels4 = labels4[l4_true]
        '''

        correct1 = (predicted1 == labels1).sum().item()
        correct2 = (predicted2 == labels2).sum().item()
        correct3 = (predicted3 == labels3).sum().item()
        correct4 = (predicted4 == labels4).sum().item()
        accuracy1 = correct1 * 100. / len(labels1)
        accuracy2 = correct2 * 100. / len(labels2)
        accuracy3 = correct3 * 100. / len(labels3)
        if len(labels4) == 0:
            accuracy4 = 0
        else:
            accuracy4 = correct4 * 100. / len(labels4)
        #accuracy = correct * 100. / len(labels)

        # print(res, accuracy)

        if train:
            self.optim1.zero_grad()
            loss1.backward()
            self.optim1.step()

            self.optim2.zero_grad()
            loss2.backward()
            self.optim2.step()

            self.optim3.zero_grad()
            loss3.backward()
            self.optim3.step()

            self.optim4.zero_grad()
            loss4.backward()
            self.optim4.step()
        
        # return loss1.item(), loss2.item(), loss3.item(), loss4.item(), accuracy1, accuracy2, accuracy3, accuracy4, \
        #        correct_count, count1+count2+count3+count4, [(logits1, labels1),(logits2, labels2),(logits3, labels3),(logits4, labels4)]
        return loss1.item(), loss2.item(), loss3.item(), loss4.item(), accuracy1, accuracy2, accuracy3, accuracy4, total_list, None
      


    def _train_epoch(self):
        self.model1.train()
        self.model2.train()
        self.model3.train()
        self.model3.train()

        losses1, losses2, losses3, losses4 = [],[],[],[]
        
        acc1, acc2, acc3, acc4 = [], [], [], []
        
        pbar1 = tqdm(self.train_data_loader)
        pbar2 = tqdm(self.train_data_loader)
        pbar3 = tqdm(self.train_data_loader)
        pbar4 = tqdm(self.train_data_loader)

        for ct, data in enumerate(pbar1):

            #loss, accuracy = self._pass(data)
            loss1, loss2, loss3, loss4, accuracy1, accuracy2, accuracy3, accuracy4 =self._pass(data)
            losses1.append(loss1)
            losses2.append(loss2)
            losses3.append(loss3)
            losses4.append(loss4)
            acc1.append(accuracy1)
            acc2.append(accuracy2)
            acc3.append(accuracy3)
            acc4.append(accuracy4)
            pbar1.set_description('[loss1: %f]' % loss1)
            pbar2.set_description('[loss2: %f]' % loss2)
            pbar3.set_description('[loss3: %f]' % loss3)
            pbar4.set_description('[loss4: %f]' % loss4)

        return np.mean(losses1), np.mean(losses2), np.mean(losses3), np.mean(losses4),\
                 np.mean(acc1), np.mean(acc2), np.mean(acc3), np.mean(acc4)

    def _val_epoch(self):
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model4.eval()

        losses1, losses2, losses3, losses4 = [],[],[],[]
        acc1, acc2, acc3, acc4, acc = [], [], [], [], []
        pbar1 = tqdm(self.val_data_loader)
        pbar2 = tqdm(self.val_data_loader)
        pbar3 = tqdm(self.val_data_loader)
        pbar4 = tqdm(self.val_data_loader)
        total_correct_count = 0
        total_count = []
        bottom_count = []

        output_dict ={}
        output_dict['levels1'] = {}
        output_dict['levels2'] = {}
        output_dict['levels3'] = {}
        output_dict['levels4'] = {}

        output_dict['levels1']['logit'] = []
        output_dict['levels1']['labels'] = []
        output_dict['levels2']['logit'] = []
        output_dict['levels2']['labels'] = []
        output_dict['levels3']['logit'] = []
        output_dict['levels3']['labels'] = []
        output_dict['levels4'] ['logit']= []
        output_dict['levels4']['labels'] = []

    

        for data in pbar1:
            loss1, loss2, loss3, loss4, accuracy1, accuracy2, accuracy3, accuracy4, total_list, bottom_list= self._pass(data, train=False)
            losses1.append(loss1)
            losses2.append(loss2)
            losses3.append(loss3)
            losses4.append(loss4)
            acc1.append(accuracy1)
            acc2.append(accuracy2)
            acc3.append(accuracy3)
            acc4.append(accuracy4)
            pbar1.set_description('[loss1: %f]' % loss1)
            pbar2.set_description('[loss2: %f]' % loss2)
            pbar3.set_description('[loss3: %f]' % loss3)
            pbar4.set_description('[loss4: %f]' % loss4)
            total_count += total_list
            #bottom_count += bottom_list
            
            
            # output_dict['levels1']['logit'] += data_list[0][0].tolist()
            # output_dict['levels1']['labels'] += data_list[0][1].tolist()

            # output_dict['levels2']['logit'] += data_list[1][0].tolist()
            # output_dict['levels2']['labels'] += data_list[1][1].tolist()

            # output_dict['levels3']['logit'] += data_list[2][0].tolist()
            # output_dict['levels3']['labels'] += data_list[2][1].tolist()

            # output_dict['levels4']['logit'] += data_list[3][0].tolist()
            # output_dict['levels4']['labels'] += data_list[3][1].tolist()
        #print(sum(total_count)/sum(bottom_count))
        result = sum(total_count)/len(total_count)
        print(result)
        import sys
        sys.exit()
  
        # np.save("/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/model_res/test_res_new.npy", output_dict)
        
        return np.mean(losses1), np.mean(losses2), np.mean(losses3), np.mean(losses4),\
                 np.mean(acc1), np.mean(acc2), np.mean(acc3), np.mean(acc4)

    def train(self):
        assert self.is_train
        for epoch in range(self.n_epochs):
            train_loss1, train_loss2, train_loss3, train_loss4, train_acc1,  \
                train_acc2, train_acc3,  train_acc4 = self._train_epoch()
            val_loss1, val_loss2, val_loss3, val_loss4, val_acc1,  \
                val_acc2, val_acc3, val_acc4 = self._val_epoch()
            self.scheduler1.step()
            self.scheduler2.step()
            self.scheduler3.step()
            self.scheduler4.step()
            print(
                '[Epoch %d/%d] [training loss1: %.5f, acc1: %.2f] [validation loss1: %.5f, acc1: %.2f] \n \
                               [training loss2: %.5f, acc2: %.2f] [validation loss2: %.5f, acc2: %.2f] \n \
                               [training loss3: %.5f, acc3: %.2f] [validation loss3: %.5f, acc3: %.2f] \n \
                               [training loss4: %.5f, acc4: %.2f] [validation loss4: %.5f, acc4: %.2f] ' %
                (epoch, self.n_epochs, train_loss1, train_acc1, val_loss1, val_acc1, \
                                train_loss2, train_acc2, val_loss2, val_acc2, \
                                train_loss3, train_acc3, val_loss3, val_acc3, \
                                train_loss4, train_acc4, val_loss4, val_acc4,)
            )

            # save model

            if epoch == self.n_epochs-1:
                save_f1 = self.model_dir + '/class1/name_best_epoch.pt'
                print('saving model to %s' % (save_f1))
                torch.save(self.model1.state_dict(), save_f1)

                save_f2 = self.model_dir + '/class2/name_best_epoch.pt'
                print('saving model to %s' % (save_f2))
                torch.save(self.model2.state_dict(), save_f2)

                save_f3 = self.model_dir + '/class3/name_best_epoch.pt'
                print('saving model to %s' % (save_f3))
                torch.save(self.model3.state_dict(), save_f3)

                save_f4 = self.model_dir + '/class4/name_best_epoch.pt'
                print('saving model to %s' % (save_f4))
                torch.save(self.model4.state_dict(), save_f4)



            # if val_acc1 < best_val_loss1:
            #     best_val_loss1 = val_loss1
            #     save_f1 = self.model_dir + '/class1/%s_best_epoch.pt' % self.type
            #     print('saving %s model to %s' % (self.type, save_f1))
            #     torch.save(self.model1.state_dict(), save_f1)

            # # save model2
            # if val_loss2 < best_val_loss2:
            #     best_val_loss2 = val_loss2
            #     save_f2 = self.model_dir + '/class2/%s_best_epoch.pt' % self.type
            #     print('saving %s model to %s' % (self.type, save_f2))
            #     torch.save(self.model2.state_dict(), save_f2)
            
            # # save model3
            # if val_loss3 < best_val_loss3:
            #     best_val_loss3 = val_loss3
            #     save_f3 = self.model_dir + '/class3/%s_best_epoch.pt' % self.type
            #     print('saving %s model to %s' % (self.type, save_f3))
            #     torch.save(self.model3.state_dict(), save_f3)
            # # savle model4
            # if val_loss4 < best_val_loss4:
            #     best_val_loss4 = val_loss4
            #     save_f4 = self.model_dir + '/class4/%s_best_epoch.pt' % self.type
            #     print('saving %s model to %s' % (self.type, save_f4))
            #     torch.save(self.model4.state_dict(), save_f4)

    def test(self):
        assert not self.is_train
        #test_loss, test_acc = self._val_epoch()
        test_loss1, test_loss2, test_loss3, test_loss4, test_acc1,  \
                test_acc2, test_acc3,  test_acc4 = self._val_epoch()
        print('[test loss1: %.2f, acc1: %.2f] \n \
               [test loss2: %.2f, acc2: %.2f]\n \
              [test loss3: %.2f, acc3: %.2f] \n \
               [test loss4: %.2f, acc4: %.2f] \n ' \
        % (test_loss1, test_acc1,test_loss2, test_acc2,test_loss3, test_acc3,test_loss4, test_acc4))
