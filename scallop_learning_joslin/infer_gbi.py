
import torch
import copy
from torch.optim import SGD

def get_lambda(model, lr=1e-3):
    model_l = copy.deepcopy(model)

    # freeze weights
    #for param in model.parameters():
    #    param.requires_grad = False

    c_opt = SGD(model_l.parameters(), lr=lr)

    return model_l, c_opt


def reg_loss(model_lambda, model, exclude_names=set()):
    orig_params = []
    lambda_params = []

    for (name, w_orig), (_, w_curr) in zip(model.named_parameters(), model_lambda.named_parameters()):
        if name not in exclude_names:
            orig_params.append(w_orig.flatten())
            lambda_params.append(w_curr.flatten())
        else:
            print('skipping %s' % name)

    orig_params = torch.cat(orig_params, dim=0)
    lambda_params = torch.cat(lambda_params, dim=0)

    return torch.linalg.norm(orig_params - lambda_params, dim=0, ord=2)


def violation_num(adj_mat, labels):
        # adj_mat is V*V, and labels is a 1*V vector , which V is number of image labels.
        V_vec = torch.matmul(labels, adj_mat) # B*E vector with {positive or zero}=no_violaton, {negative} = violation
        V_num=torch.sum(V_vec<0, 1)
        #V_avg = torch.mean(V_num.float())
        #print(f'Violation scores, and Violation average: {V_vec},  {V_avg}')
        return V_num

def run_gbi(log_probs, model, adj_mat=None, is_correct=None):
    """
    Runs gradient-based inference on program. Prints pre- and post- accuracy/constraint violations.
    data_iters: number of datapoints to test in validloader
    gbi_iters: number of gradient based inference optimization steps
    label_names: names of concepts used to get log probabilities from
    is_correct: function with parameter datanode that returns whether or not the prediction is correct
    """
    #num_satisfied_l, num_constraints_l = get_constraints(node_l)
    num_constraints_l = violation_num(adj_mat, log_probs)
    #num_satisfied_l, num_constraints_l = 2, 3
    ###Joslin: not sure what num_satisfied_l mean
    is_satisifed = 1 if num_satisfied_l == num_constraints_l else 0
    log_probs = log_probs.sum(dim=1)

    model1, model2, model3, model4 = model['name1'], model['name2'], model['name3'], model['name4']

    model_l1, c_opt = get_lambda(model1, lr=1e-1)
    model_l2, c_opt = get_lambda(model2, lr=1e-1)
    model_l3, c_opt = get_lambda(model3, lr=1e-1)
    model_l4, c_opt = get_lambda(model4, lr=1e-1)

    # constraint loss: NLL * binary satisfaction + regularization loss
    # reg loss is calculated based on L2 distance of weights between optimized model and original weights
    c_loss = -1 * log_probs * (1-is_satisifed) + reg_loss(model_l1, model1)+ reg_loss(model_l2, model2) + \
            reg_loss(model_l3, model3) + reg_loss(model_l4, model4)
    
    return c_loss

    # print("iter=%d, c_loss=%d, satisfied=%d" % (c_iter, c_loss.item(), num_satisfied_l))

    # if num_satisfied_l == num_constraints_l:
    #     satisfied = True
    #     print('SATISFIED')

    #     if is_correct(node_l):
    #         print('CORRECT')
    #     else:
    #         incorrect_after += 1

    #     break

    # c_loss.backward()
    # c_opt.step()