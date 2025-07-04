import torch
import numpy as np
import torchvision.transforms.functional as torchF
import torch.nn.functional as F
import torch.distributions as dist
import torch.nn as nn
import math
import torch
import torch.linalg as linalg


def distro_kl(x,y): # JS Divergence
    m = dist.Categorical(probs=(x+y)/2)
    p = dist.Categorical(probs=x)
    q = dist.Categorical(probs=y)
    kl_pq = dist.kl_divergence(p,m)
    kl_qp = dist.kl_divergence(q,m)
    kl = 0.5 * (kl_pq+kl_qp)
    return kl


def normalize_by_sum(tensor, dim):
    sum_along_dim = tensor.sum(dim=dim, keepdim=True)
    return tensor / sum_along_dim

def reverse_cumsum(tensor, dim):
    reversed_tensor = torch.flip(tensor, [dim])
    cumsum_reversed = torch.cumsum(reversed_tensor, dim=dim)
    return torch.flip(cumsum_reversed, [dim])


def calculate_js_cumsum_twoway(atts, args, iteration): # the js loss introduced in Appendix B
    att_map_1 = atts[args['obj1']]
    att_map_2 = atts[args['obj2']]
    direction = args['relation']
    p = -1
    if direction == 'left' or direction=="right":
        p=0
    elif direction == 'top' or direction=='bottom':
        p=1
    else:
        assert (p==1 or p==0)==True
    x1_sums = torch.sum(att_map_1, dim=p)
    s1 = torch.sum(x1_sums)
    x1_sums = x1_sums / s1
    x2_sums = torch.sum(att_map_2, dim=p)
    s2 = torch.sum(x2_sums)
    x2_sums = x2_sums/s2
    if direction == 'left' or direction =='top':
        x2_sumsfinal = x2_sums
        x1_sumsfinal = x1_sums
    else:
        x2_sumsfinal = x1_sums
        x1_sumsfinal = x2_sums
    x2_cm = torch.cumsum(x2_sumsfinal, dim=0)
    x2_cm = x2_cm / torch.sum(x2_cm)

    x1_reverse_cm = reverse_cumsum(x1_sumsfinal, dim=0)
    x1_reverse_cm = x1_reverse_cm/ torch.sum(x1_reverse_cm)
    loss = -distro_kl(x2_cm,x1_sumsfinal)-distro_kl(x1_reverse_cm,x2_sumsfinal)
    return loss



def calculate_prod_cumsum(atts, args,iteration): # our sprint loss
    att_map_1 = atts[args['obj1']]
    att_map_2 = atts[args['obj2']]
    direction = args['relation']
    p = -1 # setting the axis to perform PoS
    if direction == 'left' or direction=="right":
        p=0
    elif direction == 'top' or direction=='bottom':
        p=1
    else:
        assert (p==1 or p==0)==True
    # normalizing to make the distributions
    x1_sums = torch.sum(att_map_1, dim=p)
    s1 = torch.sum(x1_sums)
    x1_sums = x1_sums / s1
    x2_sums = torch.sum(att_map_2, dim=p)
    s2 = torch.sum(x2_sums)
    x2_sums = x2_sums/s2
    if direction == 'left' or direction =='top':
        x2_sumsfinal = x2_sums
        x1_sumsfinal = x1_sums
    else:
        x2_sumsfinal = x1_sums
        x1_sumsfinal = x2_sums
    # calculate the cdf of the distribution
    x2_cm = torch.cumsum(x2_sumsfinal, dim=0)
    # define the loss as the product
    xprod = x1_sumsfinal * x2_cm
    xprodsum = torch.sum(xprod)
    # for implemenetation purposes we used the squared loss
    loss = xprodsum*xprodsum
    return loss


def calculate_loss(attention_maps,order,i): # the loss order manager
    loss = 0
    loop = 1
    loss_types = order['loss_types']
    w=[1]
    if len(loss_types) > 1:
        w = order['weights']
        loop = len(w)

    for l in range(loop):
        loss_args = order['args'][l]
        loss_type = loss_types[l]
        loss_func = losses[loss_type]
        loss += w[l]*loss_func(atts=attention_maps,args=loss_args,iteration=i)
        
    return loss

losses = { # losses defined. You can add any loss from any paper and use it in our framework
    'ProductCumulatives' : calculate_prod_cumsum,
    'TwoSideCumJS' : calculate_js_cumsum_twoway
}