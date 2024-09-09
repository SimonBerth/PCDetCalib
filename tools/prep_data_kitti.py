# Adapted from mmdetection3d's kitti_data_utils.py
# https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/kitti_data_utils.py

from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path
import numpy as np


def get_kitti_info_path(idx,
            prefix,
            ground_truth=True,
            relative_path=True,
            exist_check=True,
            use_prefix_id=False):
    img_idx_str = '{:06d}'.format(idx) + '.txt'
    prefix = Path(prefix)
    if ground_truth:
        file_path = Path('training') / 'label_2' / img_idx_str
    else:
        file_path = Path('pred_instances_3d') / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_kitti_info(path,
                   aux_info=True,
                   ids=7481,
                   num_worker=8,
                   relative_path=True,
                   ground_truth=True,
                   clean_det=True):
    """
    KITTI annotation format version 2:
    {
        idx: int
        group_idx: [num_bbox] int array
        name: [num_bbox] str array
        bbox_3d: [num_bbox, 7] float array
        score: [num_bbox] float array
        [optional]aux_info: {
            bbox_im: [num_bbox, 4] float array
            truncated: [num_bbox] float array
            occluded: [num_bbox] int array
            alpha: [num_bbox] float array
            difficulty: [num_bbox] int array
            }
    }
    """
    root_path = Path(path)
    if not isinstance(ids, list):
        ids = list(range(ids))


    def map_func(idx):
        info = {
            'idx': idx,
            'group_idx': [],
            'name': [],
            'difficulty':None,
            'bbox_3d': [],
            'bbox_im':[],
            'score': None,
            'iou_3d' : None,
            'iou_score_3d' : None,
            'iou_bev' : None,
            'iou_score_bev' : None,
        }
        label_path = get_kitti_info_path(idx, path, relative_path=relative_path, ground_truth=ground_truth)
        if relative_path:
            label_path = str(root_path / label_path)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        content = [line.strip().split(' ') for line in lines]
        if clean_det:
            content_mask = np.array([x[0] in ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Person_sitting', 'Truck'] for x in content])
        else:
            content_mask = np.ones(len(content))
        num_objects = content_mask.sum()
        content = [line for line, val_msk in zip(content, content_mask) if val_msk]
        
        info['name'] = np.array([x[0] for x in content])
        num_bbox = len(info['name'])
        if ground_truth: #has idx
            info['group_idx'] = np.arange(num_bbox, dtype=np.int32)
        else:
            info['group_idx'] = -np.ones(num_bbox, dtype=np.int32)
        # bbox composed of loc[0:3]/dim[3:6]/rot[6] express in lhw
        info['bbox_3d'] = np.array([[float(info) for info in x[8:15]]
                for x in content]).reshape(-1, 7)[:, [3, 4, 5, 2, 0, 1, 6]]
                
        info['bbox_im'] = np.array([[float(info) for info in x[4:8]]
                                        for x in content]).reshape(-1, 4)
                                        
        if len(content) != 0 and len(content[0]) == 16:  # has score
            info['score'] = np.array([float(x[15]) for x in content])
        
        if ground_truth:
            aux_info = {}
            aux_info['truncated'] = np.array([float(x[1]) for x in content])
            aux_info['occluded'] = np.array([int(x[2]) for x in content])
            aux_info['alpha'] = np.array([float(x[3]) for x in content])
            aux_info['bbox_im'] = np.array([[float(info) for info in x[4:8]]
                                        for x in content]).reshape(-1, 4)
            info['difficulty'] = add_difficulty_to_annos(aux_info)
            if aux_info:
                info.update({
                    'aux_info':aux_info
                })
                #add_difficulty_to_annos(aux_info)
        
        return info


    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, ids)
    ret = list(image_infos)
    for i,_ in enumerate(ret):
        ret[i]['idx']=i 

    return ret
    


def add_difficulty_to_annos(annos):
    min_height = [40, 25, 25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [0, 1, 2]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [0.15, 0.3, 0.5]  # maximum truncation level of the groundtruth used for evaluation
    bbox = annos['bbox_im']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(bbox), ), dtype=bool)
    moderate_mask = np.ones((len(bbox), ), dtype=bool)
    hard_mask = np.ones((len(bbox), ), dtype=bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(bbox)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff
