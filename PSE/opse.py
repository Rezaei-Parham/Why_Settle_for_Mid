import os, json, itertools 
import numpy as np
import random
from PIL import Image
from math import comb
import numpy as np 
import itertools 
import numpy as np 
ALL_MODELS = ['Model1','Model2','Model3'] # models evaluating  

Algorithms = ['UCB'] #  algorithms considering. You can add more algorithms and implement them in the get_UCB function
MINI_BATCH = 5 # mini batch sampling during each round
reses = {}
number_of_models = 3 # Number of models to choose for each simulation
initial_samples = 5
METHOD = 'UCB'
_alpha = 2
dataset_size = 500
for i in ALL_MODELS:
    with open(f'.<scores path>.json') as p: #this was for experiment, you can run online version by combination with the segment code also provided
        reses[i] = json.load(p)

with open('<dataset path>.json', 'r') as f:
    text_data = json.load(f)

def get_UCB(mo,round_):
    global nchosen
    global METHOD
    if nchosen[mo] == 0:
        return 1e6
    mean = np.mean(visor_data[mo])
    n = np.sum(nchosen[mo])
    if METHOD == 'UCB':
        return mean + np.sqrt(np.log(round_)/n)*_alpha
    else:
        print("shat method is this?")
        assert 1==0

def get_max_UCB(round_):
    max_model = ""
    max_ucb = -9999
    for mo in np.random.permutation(ALL_MODELS):
        ucb = get_UCB(mo,round_)
        if ucb > max_ucb:
            max_ucb = ucb
            max_model = mo
    return max_model

def update_model(mo):
    global visor_data
    global all_nchosen
    vals = []
    for _ in range(MINI_BATCH):
        rchoice = random.choice(list(reses[mo].keys()))
        rchoice = reses[mo][rchoice]
        data = rchoice['score']
        vals.append(data)
    visor_data[mo].append(np.mean(vals))
    

T = 100 # total number of sampling rounds

visor_data = {}
nchosen = {}
for i in ALL_MODELS:
    nchosen[i] = []

def initial_generations():
    global ALL_MODELS
    for mo in ALL_MODELS:
        update_model(mo)
        for mod in ALL_MODELS:
            if mod == mo:
                nchosen[mod].append(1)
            else:
                nchosen[mod].append(0)

def simulate():
    global visor_data
    global ALL_MODELS
    global nchosen
    visor_data = {}
    for mo in ALL_MODELS:visor_data[mo] = []
    initial_generations()
    for t in range(initial_samples*len(ALL_MODELS)+1,T+1):
        mo = get_max_UCB(t)
        for mod in ALL_MODELS:
            if mod == mo:
                nchosen[mod].append(1)
            else:
                nchosen[mod].append(0)
        update_model(mo)
        

simulate()

np.savez(f'<path to output>.npz', **nchosen) #number of times each model is chosen

