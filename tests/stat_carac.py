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
    args = parser.parse_args()
    return args


def compute_stat(dt_data, metric_key, names='Any'):
    assert (dt_data[0]['score'] is not None), "Score values are required"
    
    prepared_data = {
        'name': [],
        'iou_metric' : [],
        'score': [],
    }
    
    nbr_det = len(dt_data)


    prepared_data['name'] = np.concatenate([ dt['name']
        for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['score'] = np.concatenate([ np.atleast_1d(dt['score'])
        for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['iou_metric'] = np.concatenate([ dt[metric_key]
        for dt in dt_data if dt['score'] is not None ], 0)
    
    if names != 'Any':
        if isinstance(names, str):
            names = [names]
        class_mask = np.isin(prepared_data['name'], names)
        for key in prepared_data:
            prepared_data[key] = prepared_data[key][class_mask]
            
    err = prepared_data['score'] - prepared_data['iou_metric']
                
    res = {
        'bias': np.mean(err),
        'variance': np.var(err),
    }
    return res



if __name__ == '__main__':
    args = parse_args()
    
    dt_path = Path(args.dt_path) / 'dt_data.pkl'

    with open(dt_path, 'rb') as dt_file:
        dt_data = pickle.load(dt_file)

    names = 'Any'
    
    res = []
    
    for m in ['iou_3d', 'iou_score_3d', 'iou_bev', 'iou_score_bev']:
        
        res_i = compute_stat(dt_data, metric_key=m, names='Any')
        res.append(res_i)
        
        print(f"For ground-truth {m}, we have bias {res_i['bias']} and variance {res_i['variance']}")
        
    
    res_path = Path(args.dt_path) / 'stat.pkl'
    with open(res_path, 'wb') as file:
        pickle.dump(res, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")
