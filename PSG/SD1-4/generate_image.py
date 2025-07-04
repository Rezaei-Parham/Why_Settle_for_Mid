from typing import List, Dict, Optional
import torch

import sys 
sys.path.append(".")
sys.path.append("..")
import numpy as np
from pipeline_sprint import AttendAndExcitePipeline
from config import RunConfig
from run import run_on_prompt, get_indices_to_alter
from utils import vis_utils
from utils.ptp_utils import AttentionStore
NUM_DIFFUSION_STEPS = 60
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

stable = AttendAndExcitePipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
tokenizer = stable.tokenizer
def run_and_display(prompts: List[str],
                    controller: AttentionStore,
                    indices_to_alter: List[int],
                    generator: torch.Generator,
                    run_standard_sd: bool = False,
                    scale_factor: int = 20,
                    max_iter_to_alter: int = 20,
                    display_output: bool = False,
                    sd_2_1: bool = False,
                    order_list: List = None,):
    config = RunConfig(prompt=prompts[0],
                       run_standard_sd=run_standard_sd,
                       scale_factor=scale_factor,
                       max_iter_to_alter=max_iter_to_alter,
                       sd_2_1=sd_2_1)
    image = run_on_prompt(model=stable,
                          prompt=prompts,
                          controller=controller,
                          token_indices=indices_to_alter,
                          seed=generator,
                          config=config,
                          order_list=order_list)
 
    return image

def generate_images_for_method(prompt: str,
                               seeds: List[int],
                               indices_to_alter: Optional[List[int]] = None,
                               is_attend_and_excite: bool = True,
                               sd_2_1: bool = False,
                               order_list: List = None
                               ,max_iter_to_alter: int = 20):
    token_indices = get_indices_to_alter(stable, prompt) if indices_to_alter is None else indices_to_alter
    images = []
    for seed in seeds:
        g = torch.Generator('cuda').manual_seed(seed)
        prompts = [prompt]
        controller = AttentionStore()
        run_standard_sd = False if is_attend_and_excite else True
        image = run_and_display(prompts=prompts,
                                controller=controller,
                                indices_to_alter=token_indices,
                                generator=g,
                                max_iter_to_alter=max_iter_to_alter,
                                run_standard_sd=run_standard_sd,
                                sd_2_1=sd_2_1,
                                order_list=order_list)
        images.append(image.resize((256, 256)))
    return images


NO_THRESHOLD = {1000:0.2}

pmp = "a dog to the left of a cat"
r = np.random.randint(0,100000)
relation = 'left'
ims = generate_images_for_method(
prompt=pmp,
seeds=[r],
indices_to_alter=[2,8],
is_attend_and_excite=False,
max_iter_to_alter=25,
order_list= [{'scale_factor':20,'thresholds':NO_THRESHOLD,'max_refinement_steps':10
            ,'loss_types':['ProductCumulatives'],'weights':[1]
            ,'args':[{'obj1':10,'obj2':1,'relation':relation}]}] 
)

img = ims[0]
img.save(f'/{pmp}_{r}.png')

