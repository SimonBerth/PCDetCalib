import os
import argparse
from pathlib import Path
import pickle
import numba
import numpy as np
from rotate_iou import rotate_iou_gpu_eval

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute IoU/IoU scores for one set of dt and gt, and put the results in the dt')
    parser.add_argument('dt_path', help='path to the folder containing dt_data.pkl')
    parser.add_argument('gt_path', help='path to the folder containing gt_data.pkl')
    args = parser.parse_args()
    return args

def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    # TODO: change to use prange for parallel mode, should check the difference
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in numba.prange(N):
        for j in numba.prange(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (
                    min(boxes[i, 1], qboxes[j, 1]) -
                    max(boxes[i, 1] - boxes[i, 4],
                        qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def calculate_ious_partly(gt_annos, dt_annos, num_parts=50):
    """
    Adapted from open-mmlab's calculate_iou_partly. Can be found in eval.py in OpenPCDet and mmdetection3d.
    Fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.

    Args:
        gt_annos: List[Dict[str, Any]], must be from prep_gt_kitti.py
        dt_annos: List[Dict[str, Any]], must be from prep_dt_kitti.py
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_iou_bev = []
    parted_iou_3d = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]

        gt_boxes_bev = np.concatenate([a["bbox_3d"][:, [0,2,3,5,6]] for a in gt_annos_part], 0)
        gt_boxes_3d = np.concatenate([a["bbox_3d"] for a in gt_annos_part], 0)
        dt_boxes_bev = np.concatenate([a["bbox_3d"][:, [0,2,3,5,6]] for a in dt_annos_part], 0)
        dt_boxes_3d = np.concatenate([a["bbox_3d"] for a in dt_annos_part], 0)
        
        iou_part_bev = rotate_iou_gpu_eval(gt_boxes_bev, dt_boxes_bev, -1).astype(np.float64)
        iou_part_3d = rotate_iou_gpu_eval(gt_boxes_bev, dt_boxes_bev, 2).astype(np.float64)
        d3_box_overlap_kernel(gt_boxes_3d, dt_boxes_3d, iou_part_3d, criterion=-1)
        iou_part_3d.astype(np.float64, copy=False)
        
        parted_iou_bev.append(iou_part_bev)
        parted_iou_3d.append(iou_part_3d)
        example_idx += num_part
        
    iou_bev = []
    iou_3d = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            iou_bev.append(
                parted_iou_bev[j][gt_num_idx:gt_num_idx + gt_box_num,
                                  dt_num_idx:dt_num_idx + dt_box_num])
            iou_3d.append(
                parted_iou_3d[j][gt_num_idx:gt_num_idx + gt_box_num,
                                 dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return iou_bev, iou_3d, total_gt_num, total_dt_num


def update_dt_iou(gt_data, dt_data, num_parts=50):
    assert(len(gt_data)==len(dt_data))

    iou_bev, iou_3d, total_gt_num, total_dt_num = calculate_ious_partly(gt_annos=gt_data, dt_annos=dt_data, num_parts=50)
        
    num_elem = len(gt_data)
    for i in range(num_elem):
        iou_3d_i = np.array(iou_3d[i])
        iou_bev_i = np.array(iou_bev[i])
        index = np.argmax(iou_3d_i,axis=0)
        
        iou_3d_i = iou_3d_i[index,range(iou_3d_i.shape[1])]
        iou_bev_i = iou_bev_i[index,range(iou_bev_i.shape[1])]
        
        dt_data[i]['iou_3d'] = iou_3d_i
        dt_data[i]['iou_score_3d'] = np.clip(2 * dt_data[i]['iou_3d'] - 0.5, 0, 1)
        dt_data[i]['iou_bev'] = iou_bev_i
        dt_data[i]['iou_score_bev'] = np.clip(2 * dt_data[i]['iou_bev'] - 0.5, 0, 1)
        
        dt_data[i]['group_idx'] = index - (dt_data[i]['iou_3d'] == 0.0)


def calculate_valid_partly(gt_annos, dt_annos, num_parts=50):
    """
    Adapted from open-mmlab's calculate_iou_partly. Can be found in eval.py in OpenPCDet and mmdetection3d.
    Fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.

    Args:
        gt_annos: List[Dict[str, Any]], must be from prep_gt_kitti.py
        dt_annos: List[Dict[str, Any]], must be from prep_dt_kitti.py
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_dt_mask_tp = []
    parted_dt_mask_fp = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        
        # Link DT and GT
        dt_len = np.array([len(a["group_idx"]) for a in dt_annos_part])
        dt_idx = np.concatenate([[a["idx"]]*l for a,l in zip(dt_annos_part,dt_len)], 0)
        dt_group_idx = np.concatenate([a["group_idx"] for a in dt_annos_part], 0)
        gt_cum_len = np.array([len(a["group_idx"]) for a in gt_annos_part])
        gt_cum_len = np.array([np.sum(gt_cum_len[:i]) for i in range(len(gt_cum_len))])
        
        #Init DT validity arrays
        dt_mask_tp = np.zeros_like(dt_group_idx,bool)
        dt_mask_ign = np.zeros_like(dt_group_idx,bool)
        dt_mask_fp = np.zeros_like(dt_group_idx,bool)
        
        #Ignore small DT
        dt_mask = ~(dt_mask_tp|dt_mask_ign|dt_mask_fp)
        dt_bbox_im = np.concatenate([a["bbox_im"] for a in dt_annos_part], 0)[dt_mask]
        dt_mask_ign[dt_mask] = np.abs(dt_bbox_im[:,3] - dt_bbox_im[:,1]) < 25
        
        #Mark unassociated and below thres DT as FP
        dt_mask = ~(dt_mask_tp|dt_mask_ign|dt_mask_fp)
        dt_mask_fp[dt_mask] = np.array(dt_group_idx[dt_mask] == -1, bool)
        
        dt_mask = ~(dt_mask_tp|dt_mask_ign|dt_mask_fp)
        dt_name = np.concatenate([a["name"] for a in dt_annos_part], 0)[dt_mask]
        name_to_class = {
            'Car': 0,
            'Pedestrian': 1,
            'Cyclist': 2,
        }
        class_thres = np.array([0.7, 0.5, 0.5])
        dt_thres = np.array([class_thres[name_to_class[name]] for name in dt_name])
        dt_iou_3d = np.concatenate([a["iou_3d"] for a in dt_annos_part], 0)[dt_mask]
        dt_mask_fp[dt_mask] = dt_iou_3d < dt_thres

        #Determine if TP, ignored, or FP depending on class of gt and dt, and diff of gt
        dt_mask = ~(dt_mask_tp|dt_mask_ign|dt_mask_fp)
        gt_to_dt = gt_cum_len[np.array(dt_idx[dt_mask],int)-example_idx] + dt_group_idx[dt_mask]
        dt_name = np.concatenate([a["name"] for a in dt_annos_part], 0)[dt_mask]
        gt_name = np.concatenate([a["name"] for a in gt_annos_part], 0)[gt_to_dt]
        gt_diff = np.concatenate([a["difficulty"] for a in gt_annos_part], 0)[gt_to_dt]

        dt_mask_ign[dt_mask] = ((((gt_name == "Van") | (gt_name == "Truck")) & (dt_name == "Car")) |((gt_name == "Person_sitting") & (dt_name == "Pedestrian")) | (gt_diff == -1))
        
        dt_mask_tp[dt_mask] = (gt_name == dt_name) & ~(dt_mask_ign[dt_mask])
        
        dt_mask_fp[dt_mask] = ~(dt_mask_tp[dt_mask] | dt_mask_ign[dt_mask])

        assert ~np.any((dt_mask_tp & dt_mask_ign)|(dt_mask_tp & dt_mask_fp)|(dt_mask_ign & dt_mask_fp))
        
        parted_dt_mask_tp.append(dt_mask_tp)
        parted_dt_mask_fp.append(dt_mask_fp)
        example_idx += num_part

    mask_tp = []
    mask_fp = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        #gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            #gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            mask_tp.append(
                parted_dt_mask_tp[j][#gt_num_idx:gt_num_idx + gt_box_num,
                                     dt_num_idx:dt_num_idx + dt_box_num])
            mask_fp.append(
                parted_dt_mask_fp[j][#gt_num_idx:gt_num_idx + gt_box_num,
                                     dt_num_idx:dt_num_idx + dt_box_num])
            #gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return mask_tp, mask_fp

def update_dt_valid(gt_data, dt_data, num_parts=50):
    assert(len(gt_data)==len(dt_data))

    mask_tp, mask_fp = calculate_valid_partly(gt_annos=gt_data, dt_annos=dt_data, num_parts=50)
        
    num_elem = len(gt_data)
    for i in range(num_elem):
        mask_tp_i = np.array(mask_tp[i])
        mask_fp_i = np.array(mask_fp[i])
        
        dt_data[i]['mask_tp'] = mask_tp_i
        dt_data[i]['mask_fp'] = mask_fp_i

        



if __name__ == '__main__':
    args = parse_args()
    
    gt_path = Path(args.gt_path) / 'gt_data.pkl'
    dt_path = Path(args.dt_path) / 'dt_data.pkl'

    
    with open(gt_path, 'rb') as gt_file:
        gt_data = pickle.load(gt_file)
    with open(dt_path, 'rb') as dt_file:
        dt_data = pickle.load(dt_file)

    update_dt_iou(gt_data=gt_data, dt_data=dt_data, num_parts=50)
    
    update_dt_valid(gt_data=gt_data, dt_data=dt_data, num_parts=50)


    with open(dt_path, 'wb') as dt_file:
        pickle.dump(dt_data, dt_file)

    print("IoUs, IoU scores, and test results (tp and fp) are computed and results saved to:", dt_path)
