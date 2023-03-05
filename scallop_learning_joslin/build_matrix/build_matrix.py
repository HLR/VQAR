import json
import numpy as np
from scipy.sparse import csr_matrix
import torch

concept_path = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/hossein_data/concepts1.json"
kb_path = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/hossein_data/hiereachy1.json"
CONCEPT_NUM = 500

with open(concept_path) as f_1, open(kb_path) as f_2:
    concepts = json.load(f_1)
    kb = json.load(f_2)



def Violation_num(adj_mat, labels):
    	# adj_mat is V*V, and labels is a 1*V vector , which V is number of image labels.
	V_vec = torch.matmul(adj_mat,labels) # positive or zero=no_violaton, negative = violation
	V_num=(V_vec<0).sum()
	V_avg = V_num/len(V_vec)
    
	print(f'Violation scores, and Violation average: {V_vec},  {V_avg}')
	return V_avg


def build_matirx():
    def adj_format(adj_mat):
        	# adj_mat is V*V tensor, which V is number of image labels.
        E = torch.count_nonzero(adj_mat)
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
        
    matrix = np.zeros((CONCEPT_NUM, CONCEPT_NUM))
    concept_dict = {}
    for id, item in enumerate(concepts):
        concept_dict[item] = id
    for i in range(len(concepts)):
        tmp_c = concepts[i]
        child = []
        for key, value in kb.items():
            if tmp_c in value:
                child = value[tmp_c]
                break
        for j in child:
            matrix[i][concept_dict[j]] = 1
    return adj_format(torch.tensor(matrix))

matrix = build_matirx()
adj_mat = adj_format(matrix)
labels = torch.zeros(500, dtype=torch.int32)	
Violation_num(adj_mat, labels)




