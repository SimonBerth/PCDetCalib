import os
import argparse
from pathlib import Path
import pickle
import numpy as np
from datasets.prep_gt_kitti import get_gt_kitti
from results.prep_dt_kitti import get_dt_kitti
from tools.update_dt import update_dt_iou, update_dt_valid
import tests.ECE as ECE
import tests.ECE_regr as ECE_regr
import tests.NMS_qual as NMS_qual
import tests.stat_carac as stat_carac
import tests.spars as spars






def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute IoU/IoU scores for one set of dt and gt, and put the results in the dt')
    parser.add_argument('dt_path', help='path to the folder containing dt_data.pkl')
    parser.add_argument('gt_path', nargs='?', default=Path(__file__).parent / 'datasets/kitti', help='path of the folder containing training and ImageSets for Kitti')
    parser.add_argument('--im_path', default=None, help='path of the folder containing the images')
    parser.add_argument('--relative_path', type=bool, default=True, help='wether the given paths are relative or absolute')
    parser.add_argument('--no_update', action='store_true', help='avoid updating dt_data.pkl and gt_data.pkl when running')
    parser.add_argument( '--use_index', type=bool, default=True, help='recommended True if OpenPCDet, False if mmdetection3d')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.no_update:
        gt_path = Path(args.gt_path) / 'gt_data.pkl'
        with open(gt_path, 'rb') as file:
            gt_data = pickle.load(file)
        abs_gt_path = os.path.abspath(gt_path)
        print(f"GT data has been loaded from {abs_gt_path}")
        
        dt_path = Path(args.dt_path) / 'dt_data.pkl'
        with open(dt_path, 'rb') as file:
            dt_data = pickle.load(file)
        abs_dt_path = os.path.abspath(dt_path)
        print(f"DT data has been loaded from {abs_dt_path}")
    
    else:
        gt_data = get_gt_kitti(args.gt_path, args.relative_path, clean_det=True)
        gt_path = Path(args.gt_path) / 'gt_data.pkl'
        with open(gt_path, 'wb') as file:
            pickle.dump(gt_data, file)
        abs_gt_path = os.path.abspath(gt_path)
        print(f"GT data has been saved to {abs_gt_path}")
        
        dt_data = get_dt_kitti(args.dt_path, args.relative_path, use_index=args.use_index)
        update_dt_iou(gt_data=gt_data, dt_data=dt_data, num_parts=50)
        update_dt_valid(gt_data=gt_data, dt_data=dt_data, num_parts=50)
        dt_path = Path(args.dt_path) / 'dt_data.pkl'
        with open(dt_path, 'wb') as file:
            pickle.dump(dt_data, file)
        abs_dt_path = os.path.abspath(dt_path)
        print(f"DT data has been saved to {abs_dt_path}, with IoU and validity computed")
      
    #Test parameters
    splits_ECE = np.linspace(0, 1, 11)
    splits_ECE_regr = np.linspace(0, 1, 11)
    size_bins_NMS_qual = 10
    num_bins_spars = 100
    names = 'Any'
    
    if args.im_path != None:
        im_path = Path(args.im_path)
    else:
        if isinstance(names, list):
            tmp = None
            for n in names:
                tmp = tmp + n
            names = tmp
        fold_name = 'imres_' + names +'/'
        im_path = Path(args.dt_path) / fold_name
        os.makedirs(im_path, exist_ok=True)
    
    #Computation of the ECE
    prepared_data = ECE.prepare_data_ECE(dt_data=dt_data, splits_lim=splits_ECE, names=names)
    res = ECE.compute_ECE(prepared_data)
    ECE.plot_acc_metrics(res=res, im_path=im_path, splits_lim=splits_ECE)
    res_path = Path(args.dt_path) / 'ECE.pkl'
    with open(res_path, 'wb') as file:
        pickle.dump(res, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")
    
    #Computation of the ECE adapted for regressor
    prepared_data = ECE_regr.prepare_data_ECE_regr(dt_data=dt_data, splits_lim=splits_ECE_regr, names=names)
    res=[]
    for m in ['iou_3d', 'iou_score_3d', 'iou_bev', 'iou_score_bev']:
        res_i = ECE_regr.compute_ECE_regr(prepared_data, metric=m)
        res.append(res_i)
        ECE_regr.plot_iou_metrics(res=res_i, im_path=im_path, splits_lim=splits_ECE_regr)
    res_path = Path(args.dt_path) / 'ECE_regr.pkl'
    with open(res_path, 'wb') as file:
        pickle.dump(res, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")
        
    #Computation of the pseudo sparsification curbe
    res = []
    for m in ['iou_3d', 'iou_score_3d', 'iou_bev', 'iou_score_bev']:
        prepared_data_bins = NMS_qual.prepare_data_NMS_eval(dt_data=dt_data, metric_key=m, names=names, size_bins=size_bins_NMS_qual)
        res_i = NMS_qual.compute_NMS_eval(prepared_data_bins, bin_thr=1)
        res.append(res_i)
        NMS_qual.plot_NMS_eval_metrics(res=res_i, im_path=im_path)
    res_path = Path(args.dt_path) / 'NMS_eval.pkl'
    with open(res_path, 'wb') as file:
        pickle.dump(res, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")

    #Computation of classical estimator quality metrics
    res = []
    for m in ['iou_3d', 'iou_score_3d', 'iou_bev', 'iou_score_bev']:
        res_i = stat_carac.compute_stat(dt_data, metric_key=m, names='Any')
        res.append(res_i)
        print(f"For ground-truth {m}, we have bias {res_i['bias']} and variance {res_i['variance']}")
    res_path = Path(args.dt_path) / 'stat.pkl'
    with open(res_path, 'wb') as file:
        pickle.dump(res, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")
    
    #sparsification computation
    res = spars.compute_spars(dt_data, names=names, num_bins = num_bins_spars)
    spars.plot_spars(res=res, im_path=im_path)
    res_path = Path(args.dt_path) / 'spars.pkl'
    with open(res_path, 'wb') as file:
        pickle.dump(res, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")
