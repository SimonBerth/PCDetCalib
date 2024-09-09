import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import numba
from rotate_iou import rotate_iou_gpu_eval

'''
def parse_args():
    parser = argparse.ArgumentParser(
        description='Format the given results for Kitti val')
    parser.add_argument('res_path', help='path of the folder containing pred_instances_3d')
    parser.add_argument(
        '--relative_path', type=bool, default=True,
        help='wether the given path is relative or absolute')
    args = parser.parse_args()
    return args
'''

def load_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


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
        
        iou_part_bev = rotate_iou_gpu_eval(gt_boxes_bev, dt_boxes_bev, 2).astype(np.float64)
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

def update_iou(gt_data, dt_data, num_parts=50):
    assert(len(gt_data)==len(dt_data))

    iou_bev, iou_3d, total_gt_num, total_dt_num = calculate_ious_partly(gt_annos=gt_data, dt_annos=dt_data, num_parts=50)
        
    num_elem = len(gt_data)
    for i in range(10):
    
        test = np.array(iou_3d[i])
        ind = np.argmax(test,axis=0)
        
        print("N:", i)
        print("GT:", len(gt_data[i]['bbox_3d']))
        print("DT:", len(dt_data[i]['bbox_3d']))
        print("IOU:", test)
        print("MAX:", test[ind,range(test.shape[1])], ind)
    
    return iou_bev,iou_3d






if __name__ == '__main__':
    #args = parse_args()

    gt_path = "../datasets/kitti/gt_data.pkl"
    dt_path = "../results/pv_rcnn_kitti_results/dt_data.pkl"
    
    gt_data = load_data(gt_path)
    dt_data = load_data(dt_path)
    
    print(len(gt_data))
    print(len(dt_data))


    update_iou(gt_data=gt_data, dt_data=dt_data, num_parts=50)

    
    print(f"The first element in {os.path.abspath(gt_path)} is:")
    #print(gt_data[0:10])

    print(f"\nThe first element in {os.path.abspath(dt_path)} is:")
    #print(dt_data[2:4])
    
    

