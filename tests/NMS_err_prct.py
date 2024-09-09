import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os



def parse_args():
    parser = argparse.ArgumentParser(
        description='Format the given results for Kitti val')
    parser.add_argument('dt_path_sans_nms', help='path of the folder containing pred_instances_3d for detection without nms')
    parser.add_argument('dt_path_with_nms', help='path of the folder containing pred_instances_3d for detection with nms')
    parser.add_argument('im_path', nargs='?', default=None, help='path of the folder containing the images')
    args = parser.parse_args()
    return args



def compute_pct_err(dt_data_nms, dt_data_nmsfree):
    assert(len(dt_data_nms) == len(dt_data_nmsfree))
    
    tot_gt_det_pre_nms = 0
    tot_gt_det_post_nms = 0

    for i, elem_free in enumerate(dt_data_nmsfree):
        elem_nms = dt_data_nms[i]
        
        mask_free = elem_free['mask_tp'] ^ elem_free['mask_fp']
        mask_nms = elem_nms['mask_tp'] ^ elem_nms['mask_fp']

        for j in range(np.max(elem_free['group_idx'])):
            index_gt_free = np.where(elem_free['group_idx'][mask_free] == j)[0]
            index_gt_nms = np.where(elem_nms['group_idx'][mask_nms] == j)[0]

            gt_det_free = np.any(elem_free['mask_tp'][mask_free][index_gt_free])
            gt_det_nms = np.any(elem_nms['mask_tp'][mask_nms][index_gt_nms])

            tot_gt_det_pre_nms += gt_det_free
            tot_gt_det_post_nms += gt_det_nms
            
            if (gt_det_nms and not gt_det_free):
                print(elem_free)
                print(elem_nms)
            
            assert not (gt_det_nms and not gt_det_free)
            
    res = 1 - tot_gt_det_post_nms/tot_gt_det_pre_nms
        
    return res
    
    

if __name__ == '__main__':
    args = parse_args()
    
    dt_path_with_nms = Path(args.dt_path_with_nms) / 'dt_data.pkl'
    dt_path_sans_nms = Path(args.dt_path_sans_nms) / 'dt_data.pkl'

    with open(dt_path_with_nms, 'rb') as dt_file:
        dt_data_with_nms = pickle.load(dt_file)
    with open(dt_path_sans_nms, 'rb') as dt_file:
        dt_data_sans_nms = pickle.load(dt_file)
    
    prct_err = compute_pct_err(dt_data_nms=dt_data_with_nms, dt_data_nmsfree=dt_data_sans_nms)
    
    print("Pourcentage of GT no longer detected due to bad score pre-NMS (as a % of all correctly detected GT pre-NMS) :", prct_err, "%")

    res_path = Path(args.dt_path_sans_nms) / 'NMS_err_prct.pkl'
    with open(res_path, 'wb') as file:
        pickle.dump(prct_err, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")


