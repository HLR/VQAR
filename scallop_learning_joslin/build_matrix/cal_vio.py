import torch
import numpy as np
from scipy.sparse import csr_matrix
  
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
	

def Violation_num(adj_mat, labels):
	# adj_mat is V*V, and labels is a 1*V vector , which V is number of image labels.
	V_vec = torch.matmul(adj_mat,labels) # positive or zero=no_violaton, negative = violation
	V_num=(V_vec<0).sum()
	V_avg = V_num/len(V_vec)
    
	print(f'Violation scores, and Violation average: {V_vec},  {V_avg}')
	return V_avg
	


##test sample
row = np.array([0, 1])
col = np.array([1, 2])
data = np.array([1, 1])
B = csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
B = torch.from_numpy(B)

C = adj_format(B)	
P = torch.tensor([2,1,0],dtype=torch.int32)
Violation_num(C, P)

P = torch.tensor([0,1,2],dtype=torch.int32)
Violation_num(C, P)

