import numpy as np
import matplotlib.pyplot as plt
import json
import torch

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
    # calculate the cdf of the distribution
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
    if np.sum(xmask1) == 0:
        return 0,'no-obj1'
    if np.sum(xmask2) == 0:
        return 0,'no-obj2'
    relation, reverse_relation = get_rel_pair(rel)
    if relation in ['left','right']:
        loss = calculate_prod_cumsum(xmask1/np.sum(xmask1),xmask2/np.sum(xmask2),relation)
        rev_loss = calculate_prod_cumsum(xmask1/np.sum(xmask1),xmask2/np.sum(xmask2),reverse_relation)
    elif relation in ['top','bottom']:
        loss = calculate_prod_cumsum(ymask1/np.sum(ymask1),ymask2/np.sum(ymask2),relation)
        rev_loss = calculate_prod_cumsum(ymask1/np.sum(ymask1),ymask2/np.sum(ymask2),reverse_relation)
    return max((rev_loss-loss).item(),0),(loss.item(),rev_loss.item())
    
model = '<model name>'
dic = dict(np.load(f'./<path>.npz'))


f = open('<dataset path>.json')
js = json.load(f)
scores = {}
it = 0
scoresum = 0
score_thresh = 0
both_present = 0
for instruct in js:
    it += 1
    uid = instruct['unique_id']
    text= instruct['text']
    xmask1,ymask1,xmask2,ymask2 = dic[f'{uid}']
    relation = instruct['rel_type']
    score,detail = get_score(xmask1,ymask1,xmask2,ymask2,relation)
    scores[uid] = {'score':score,'detail':detail,'text':text}
    if len(detail) == 2:
        print(detail)
        both_present += 1
    scoresum += score
    if score > 0.01:
        score_thresh += 1
print("mean score:",scoresum/it)
print('with thresh:',score_thresh/it)
print('both present:',both_present/it)
with open(f'./<output path>.json','w') as outfile:
    json.dump(scores,outfile,indent=4)