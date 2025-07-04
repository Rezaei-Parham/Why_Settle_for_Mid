import torch
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
import torch.distributions as dist


def find_com(att_map):
    c_x, c_y = 0, 0
    for i in range(att_map.shape[0]):
        c_x += att_map[i,:] * i
    for j in range(att_map.shape[1]):
        c_y += att_map[:,j] * j
    c_x, c_y = c_x.sum(), c_y.sum()
    return [c_x, c_y]

def calculate_com_distance(att_map_1, att_map_2):
    att_map_1 = att_map_1 / att_map_1.sum()
    att_map_2 = att_map_2 / att_map_2.sum()
    com_1, com_2 = find_com(att_map_1), find_com(att_map_2)
    return torch.sqrt(((com_1[0] - com_2[0]) * (com_1[0] - com_2[0])) + ((com_1[1] - com_2[1]) * (com_1[1] - com_2[1])))

def calculate_mean_variance(att_map_1, att_map_2):
    att_map_1 = att_map_1 / att_map_1.sum()
    att_map_2 = att_map_2 / att_map_2.sum()
    com_1, com_2 = find_com(att_map_1), find_com(att_map_2) 
    variance_1, variance_2 = 0, 0
    for i in range(att_map_1.shape[0]):
        for j in range(att_map_1.shape[1]):
            variance_1 += att_map_1[i][j] * ((com_1[0] - i) * (com_1[0] - i) + (com_1[1] - j) * (com_1[1] - j))
            variance_2 += att_map_2[i][j] * ((com_2[0] - i) * (com_2[0] - i) + (com_2[1] - j) * (com_2[1] - j))
    return (variance_1 + variance_2) / 2


def calculate_min_intensity(att_map_1, att_map_2):
    intensity_1 = att_map_1.sum() / (att_map_1.shape[0] * att_map_1.shape[1]) # att_map_1.mean()
    intensity_2 = att_map_2.sum() / (att_map_2.shape[0] * att_map_2.shape[1])
    return torch.min(intensity_1, intensity_2)

def calculate_overlap_ps(att_map_1, att_map_2):
    att_map_1 = att_map_1 / att_map_1.sum()
    att_map_2 = att_map_2 / att_map_2.sum()
    return ((att_map_1 * att_map_2) / (att_map_1 + att_map_2)).sum()

def calculate_cc(att_map_1, att_map_2):
    att_map_1 = att_map_1 / att_map_1.sum()
    att_map_2 = att_map_2 / att_map_2.sum()
    att_map_mean = torch.max(att_map_1, att_map_2)
    return att_map_mean.sum() / (att_map_mean.shape[0] * att_map_mean.shape[1]) # mean

def calculate_mean_kl_divergences(att_map_1, att_map_2):
    att_map_1 /= att_map_1.sum()
    att_map_2 /= att_map_2.sum()
    return (torch.sum(att_map_1 * torch.log(att_map_1 / att_map_2)) + torch.sum(att_map_2 * torch.log(att_map_2 / att_map_1))) / 2


def compute_loss(attention_maps,
                 indices_to_alter,
                 direction,
                 multiplier,
                 i) -> torch.Tensor:
    att_map_1 = attention_maps[:, :, indices_to_alter[0]]
    att_map_2 = attention_maps[:, :, indices_to_alter[1]]
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
    xprod = x1_sumsfinal * x2_cm
    xprodsum = torch.sum(xprod)
    loss = xprodsum*xprodsum
    return multiplier * loss


def compute_loss_multi(attention_maps,
                 indices_to_alter,
                 directions,
                 multiplier,
                 i) -> torch.Tensor:
    loss = torch.empty(0, device='cuda')
    for dir_idx in range(len(directions)):
        direction = directions[dir_idx]
        indices = indices_to_alter[dir_idx]
        att_map_1 = attention_maps[:, :, indices[0]]
        att_map_2 = attention_maps[:, :, indices[1]]
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
        xprod = x1_sumsfinal * x2_cm
        xprodsum = torch.sum(xprod)
        los2 = xprodsum*xprodsum
        loss = torch.cat((loss, los2.unsqueeze(0)), dim=0)
    loss = torch.sum(loss)
    print(loss)
    return multiplier * loss


def reverse_cumsum(tensor, dim):
    reversed_tensor = torch.flip(tensor, [dim])
    cumsum_reversed = torch.cumsum(reversed_tensor, dim=dim)
    return torch.flip(cumsum_reversed, [dim])


def distro_kl(x,y):
    x, y = x/torch.sum(x), y/torch.sum(y)
    m = dist.Categorical(probs=(x+y)/2)
    p = dist.Categorical(probs=x)
    q = dist.Categorical(probs=y)
    kl_pq = dist.kl_divergence(p,m)
    kl_qp = dist.kl_divergence(q,m)
    kl = 0.5 * (kl_pq+kl_qp)
    return kl


