
import abc
import torch
from IPython.display import display
from typing import List
from diffusers.models.attention_processor import Attention
from typing import List
from PIL import Image
import numpy as np
import os


class CrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    # def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, 1)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def register_attention_control(model, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = CrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0

def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    # print(num_pixels)
    for location in from_where:
        # print(location, attention_maps.keys())
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            # print(item.shape)
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
   

    return out

def find_com(att_map):
    c_x, c_y = 0, 0
    att_map = att_map/torch.sum(att_map)
    for i in range(att_map.shape[0]):
        c_x += att_map[i,:] * i
    for j in range(att_map.shape[1]):
        c_y += att_map[:,j] * j
    c_x, c_y = c_x.sum(), c_y.sum()
    return [c_x, c_y]

def find_var(att_map):
    v_x, v_y = 0, 0
    att_map = att_map/torch.sum(att_map)
    for i in range(att_map.shape[0]):
        v_x += att_map[i,:] * i * i
    for j in range(att_map.shape[1]):
        v_y += att_map[:,j] * j * j
    v_x, v_y = v_x.sum(), v_y.sum()
    c_x, c_y = find_com(att_map)
    v_x = v_x - c_x * c_x
    v_y = v_y - c_y * c_y
    return [v_x, v_y]


def sprint_value(att1,att2,dir='left'):
    att_map_1 = att1
    att_map_2 = att2
    direction = dir
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
    return loss

def get_1d(attention):
    x = torch.sum(attention,dim=0)
    x = x/torch.sum(x)
    return x

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
def get_plot(x1_np,path,is_green):
    indices = np.arange(len(x1_np))

    # Create smooth curves using spline interpolation
    x_smooth = np.linspace(indices.min(), indices.max(), 500)
    spl1 = make_interp_spline(indices, x1_np, k=3)
    y1_smooth = spl1(x_smooth)

    # Plot the smooth density curves and fill the area under each curve
    plt.figure(figsize=(8, 4))
    col = 'lightcoral'
    if is_green:
        col = 'SeaGreen'
    plt.plot(x_smooth, y1_smooth, color=col, lw=2, label='Tensor 1')
    plt.fill_between(x_smooth, y1_smooth, color=col, alpha=0.4)
    # Customize the axes
    plt.tick_params(left=False, bottom=False)  # Remove tick marks
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks([])  # Remove x-axis numbers
    plt.yticks([])  # Remove y-axis numbers

    # Save the plot as an image with a transparent background
    plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)

   

def save_attention_maps(attention_maps, indices_to_alter, prompt, timestep, output_path, seed=42):
    os.makedirs(f"{output_path}/{prompt}/{int(seed)}/", exist_ok=True)
