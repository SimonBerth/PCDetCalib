import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os



def parse_args():
    parser = argparse.ArgumentParser(
        description='Format the given results for Kitti val')
    parser.add_argument('dt_path', help='path of the folder containing pred_instances_3d')
    #parser.add_argument('gt_path', help='path to the folder containing gt_data.pkl')
    parser.add_argument('im_path', nargs='?', default=None, help='path of the folder containing the images')
    args = parser.parse_args()
    return args


def prepare_data_NMS_eval(dt_data, metric_key, names='Any', size_bins = 20):
    assert (dt_data[0]['score'] is not None), "Score values are required"
    assert (dt_data[0]['iou_3d'] is not None), "IoU precomputation is required"
    assert (dt_data[0]['iou_score_3d'] is not None), "IoU precomputation is required"
    assert (dt_data[0]['iou_bev'] is not None), "IoU precomputation is required"
    assert (dt_data[0]['iou_score_bev'] is not None), "IoU precomputation is required"
    assert (dt_data[0]['mask_tp'] is not None), "TP precomputation is required"
    assert (dt_data[0]['mask_fp'] is not None), "FP precomputation is required"
    
    prepared_data = {
        'idx': [],
        'group_idx': [],
        'name' : [],
        'iou_metric' : [],
        'score': [],
        'mask_tp' : [],
        'mask_fp' : [],
    }
    
    nbr_det = len(dt_data)
    prepared_data_bins ={}

    prepared_data['idx'] = np.concatenate([ [dt['idx']]*len(dt['group_idx'])
        for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['group_idx'] = np.concatenate([ dt['group_idx']
        for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['name'] = np.concatenate([ dt['name']
        for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['score'] = np.concatenate([ np.atleast_1d(dt['score'])
        for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['iou_metric'] = np.concatenate([ dt[metric_key]
        for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['mask_tp'] = np.concatenate([ dt['mask_tp']
        for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['mask_fp'] = np.concatenate([ dt['mask_fp']
        for dt in dt_data if dt['score'] is not None ], 0)

    #sort_order = np.argsort(prepared_data['score'])[::-1]
    #for key in prepared_data[metric]:
    #    prepared_data[metric][key] = prepared_data[key][sort_order]

    sort_order = np.argsort(prepared_data['iou_metric'])[::-1]
    for key in prepared_data:
        if key != 'metric':
            prepared_data[key] = prepared_data[key][sort_order]
    
    if names != 'Any':
        if isinstance(names, str):
            names = [names]
        class_mask = np.isin(prepared_data['name'], names)
        for key in prepared_data:
            prepared_data[key] = prepared_data[key][class_mask]
                
    ign_mask = prepared_data['mask_tp'] | prepared_data['mask_fp']
    for key in prepared_data:
        prepared_data[key] = prepared_data[key][ign_mask]
    
    num_elem = len(prepared_data['idx'])
    q = num_elem // size_bins
    r = num_elem % size_bins
    bin_sizes = [size_bins]*q + [r]*(r>0)
    
    prepared_data_bins = []
    bin_start = 0
    for i in range(len(bin_sizes)):
        prepared_data_bins.append({
            'metric':metric_key,
            'idx': prepared_data['idx'][bin_start:bin_start+bin_sizes[i]],
            'group_idx': prepared_data['group_idx'][bin_start:bin_start+bin_sizes[i]],
            'name': prepared_data['name'][bin_start:bin_start+bin_sizes[i]],
            'iou_metric': prepared_data['iou_metric'][bin_start:bin_start+bin_sizes[i]],
            'score': prepared_data['score'][bin_start:bin_start+bin_sizes[i]],
            'mask_tp': prepared_data['mask_tp'][bin_start:bin_start+bin_sizes[i]],
            'mask_fp': prepared_data['mask_fp'][bin_start:bin_start+bin_sizes[i]],
        })
        bin_start += bin_sizes[i]

            
    return prepared_data_bins
           
           
def compute_NMS_eval(prepared_data_bins, bin_thr=1):
    res = {
        'metric':prepared_data_bins[0]['metric'],
        'MAE':None,
        'm_score':[],
        'm_iou_metric':[],
    }
        
    data = prepared_data_bins
    
    res['MAE'] = np.mean(np.abs(np.concatenate([
            bin['score'] - bin['iou_metric'] for bin in prepared_data_bins], 0)))
        
    for i in range(len(prepared_data_bins)):
        if len(prepared_data_bins[i])>=bin_thr:
            res['m_score'].append(np.mean(prepared_data_bins[i]['score']))
            res['m_iou_metric'].append(np.mean(prepared_data_bins[i]['iou_metric']))
    return res

    
    
def plot_NMS_eval_metrics(res, im_path=None):

    metric_key = res['metric']
    m_score = res['m_score']
    m_iou_metric = res['m_iou_metric']
    index = np.arange(len(m_score))/len(m_score)

    plt.figure(figsize=(10, 6))
    
    #plt.plot(index, m_score, label='Score', linestyle='-', color='red')
    plt.scatter(index, m_score, label='Score',s=1, color='red')
    plt.plot(index, m_iou_metric, label='IoU Metric', linestyle='--', color='green')
    
    MAE = res['MAE']
    plt.title(f'Score and IoU Metric over Normalised Index (MAE: {MAE})')
    plt.xlabel('Normalised index')
    plt.ylabel('Values')
    plt.legend()
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    file_name = f"{metric_key}_NMS_eval.png"
    save_path = os.path.join(im_path, file_name)
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")
    
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    
    dt_path = Path(args.dt_path) / 'dt_data.pkl'

    with open(dt_path, 'rb') as dt_file:
        dt_data = pickle.load(dt_file)

    names = 'Any'
    size_bins = 20
    
    
    res_path = Path(args.dt_path) / 'NMS_eval.pkl'
    
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
    
    res = []
    
    for m in ['iou_3d', 'iou_score_3d', 'iou_bev', 'iou_score_bev']:
        prepared_data_bins = prepare_data_NMS_eval(dt_data=dt_data, metric_key=m, names=names, size_bins=size_bins)
        
        res_i = compute_NMS_eval(prepared_data_bins, bin_thr=1)
        res.append(res_i)
    
        plot_NMS_eval_metrics(res=res_i, im_path=im_path)
        
    with open(res_path, 'wb') as file:
        pickle.dump(res, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")
