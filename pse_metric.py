
import torch
def get_projected_mask(mask):
    xmask = torch.sum(mask, dim=0)
    xmask = (xmask/torch.sum(xmask))
    ymask = torch.sum(mask, dim=1)
    ymask = (ymask/torch.sum(ymask))
    return xmask, ymask

def calculate_prod_cumsum(mask1,mask2,relation): # PSG LOSS
    x1_sums = torch.tensor(mask1)
    x2_sums = torch.tensor(mask2)
    direction = relation
    
    if direction == 'left' or direction =='top':
        x2_sumsfinal = x2_sums
        x1_sumsfinal = x1_sums
    else:
        x2_sumsfinal = x1_sums
        x1_sumsfinal = x2_sums
    x2_cm = torch.cumsum(x2_sumsfinal, dim=0)
    xprod = x1_sumsfinal * x2_cm
    xprodsum = torch.sum(xprod)
    loss = xprodsum
    return loss

def get_rel_pair(rel):
    if rel == 'to the right of':
        return 'right','left'
    elif rel == 'to the left of':
        return 'left','right'
    elif rel == 'above':
        return 'top','bottom'
    elif rel =='below':
        return 'bottom','top'
    print("undefined")

def get_score(xmask1,ymask1,xmask2,ymask2,rel):
    if torch.sum(xmask1) == 0:
        return 0,'no-obj1'
    if torch.sum(xmask2) == 0:
        return 0,'no-obj2'
    relation, reverse_relation = get_rel_pair(rel)
    if relation in ['left','right']:
        loss = calculate_prod_cumsum(xmask1/torch.sum(xmask1),xmask2/torch.sum(xmask2),relation)
        rev_loss = calculate_prod_cumsum(xmask1/torch.sum(xmask1),xmask2/torch.sum(xmask2),reverse_relation)
    elif relation in ['top','bottom']:
        loss = calculate_prod_cumsum(ymask1/torch.sum(ymask1),ymask2/torch.sum(ymask2),relation)
        rev_loss = calculate_prod_cumsum(ymask1/torch.sum(ymask1),ymask2/torch.sum(ymask2),reverse_relation)
    return max((rev_loss-loss).item(),0),(loss.item(),rev_loss.item())
    

def get_score_from_mask(mask1,mask2,rel):
    xmask1,ymask1 = get_projected_mask(mask1)
    xmask2,ymask2 = get_projected_mask(mask2)
    return get_score(xmask1,ymask1,xmask2,ymask2,rel)