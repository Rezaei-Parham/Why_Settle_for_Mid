import os
HOME = os.getcwd()
import sys
import supervision as sv
# Folow the instruction of GroundingDINO, afterwards run the following code
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from GroundingDINO.groundingdino.util.inference import Model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
SAM_ENCODER_VERSION = "vit_h"
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)
from typing import List
import cv2
import supervision as sv
import numpy as np
from segment_anything import SamPredictor

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"{class_name}"
        for class_name
        in class_names
    ]

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def get_mask_obj(detections,titles,name):
  mask = np.zeros(shape=detections.mask[0].shape)
  for i in range(len(titles)):
    if titles[i] == name:
      mask += detections.mask[i]
  mask = (mask >= 1)
  return mask

zero_array = np.array([0 for _ in range(1024)])

def get_masks(img_path,classes,bthresh=0.45):
    SOURCE_IMAGE_PATH = img_path
    BOX_TRESHOLD = bthresh
    TEXT_TRESHOLD = 0.25
    image = cv2.imread(SOURCE_IMAGE_PATH)

    CLASSES = [classes[0]]
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    print(CLASSES)
    cids = [class_id for _, _, confidence, class_id, _
        in detections]
    print(cids)
    if 0 in cids:
      detections.mask = segment(
          sam_predictor=sam_predictor,
          image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
          xyxy=detections.xyxy
      )
      titles = [
      CLASSES[class_id]
      for class_id
      in detections.class_id]

      obj1 = torch.tensor(get_mask_obj(detections,titles,CLASSES[0]))
      obj1_xmask = torch.sum(obj1, dim=0)
      obj1_xmask = (obj1_xmask/torch.sum(obj1_xmask)).numpy()
      obj1_ymask = torch.sum(obj1, dim=1)
      obj1_ymask = (obj1_ymask/torch.sum(obj1_ymask)).numpy()
    else:
      obj1_xmask = zero_array
      obj1_ymask = zero_array

    CLASSES = [classes[1]]
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    print(CLASSES)
    cids = [class_id for _, _, confidence, class_id, _
        in detections]
    print(cids)
    if 0 in cids:
      detections.mask = segment(
          sam_predictor=sam_predictor,
          image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
          xyxy=detections.xyxy
      )
      titles = [
      CLASSES[class_id]
      for class_id
      in detections.class_id]

      obj2 = torch.tensor(get_mask_obj(detections,titles,CLASSES[0]))
      obj2_xmask = torch.sum(obj2, dim=0)
      obj2_xmask = (obj2_xmask/torch.sum(obj2_xmask)).numpy()
      obj2_ymask = torch.sum(obj2, dim=1)
      obj2_ymask = (obj2_ymask/torch.sum(obj2_ymask)).numpy()
    else:
      obj2_xmask = zero_array
      obj2_ymask = zero_array
    return obj1_xmask,obj1_ymask,obj2_xmask,obj2_ymask


import json
f = open('<ADDRESS HERE>')
js = json.load(f)
name = "<NAME OF THE MODEL>"
print("model == ",name)
thresh = 0.35
SHAPE0 = 1024
masks_arr = {}
for i in js:
    uniq_id = i['unique_id']
    obj1 = i['obj_1_attributes'][0]
    obj2 = i['obj_2_attributes'][0]
    obj_masks = get_masks(f"<IMAGES PATH>.png",[obj1,obj2],bthresh=thresh)
    masks_arr[f'{uniq_id}'] = obj_masks

json.dump(masks_arr,open('./<NAME OF OUTPUT>.json','w'))
np.savez(f'./{name}_masks_{thresh}.npz',**masks_arr)