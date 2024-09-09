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
    parser.add_argument('im_path', nargs='?', default=None, help='path of the folder containing the images')
    args = parser.parse_args()
    return args


def prepare_data_ECE_regr(dt_data, splits_lim = np.linspace(0, 1, 11), names='Any'):
    """
    names : str, list[str], or 'Any' for all detection regardless of class
    """
    assert (dt_data[0]['score'] is not None), "Score values are required"

    prepared_data = {
        'idx': [],
        'group_idx': [],
        'score': [],
    }
        
    
    nbr_det = len(dt_data)
    prepared_data['idx'] = np.concatenate(
        [ [dt['idx']]*len(dt['group_idx']) for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['score'] = np.concatenate(
        [ np.atleast_1d(dt['score']) for dt in dt_data if dt['score'] is not None ], 0)
    for key in dt_data[0]:
        if key not in ['idx','score','difficulty']:
            prepared_data[key] = np.concatenate(
                [dt[key] for dt in dt_data if dt['score'] is not None], 0)
      
    sort_order = np.argsort(prepared_data['score'])
    for key in prepared_data:
        prepared_data[key] = prepared_data[key][sort_order]
        
    if names != 'Any':
        if isinstance(names, str):
            names = [names]
        class_mask = np.isin(prepared_data['name'], names)
        for key in prepared_data:
            prepared_data[key] = prepared_data[key][class_mask]
        
    bin_inds = np.digitize(prepared_data['score'], splits_lim[1:], right=True)

    prepared_data_bins = []
    for i in range(len(splits_lim)-1):
        bin_msk = (bin_inds==i)
        tmp = {}
        for key in prepared_data:
            tmp[key] = prepared_data[key][bin_msk]
        prepared_data_bins.append(tmp)
                    
    return prepared_data_bins


def compute_ECE_regr(prepared_data_bins, metric='iou_score_3d', bin_thr=1):
    '''
    metric : name of the studied metric
    bin_thr : min number of element for a bin to be used
    '''
    
    
    res = {
        'metric':metric,
        'ECE_regr':None,
        'ECE_regr_unweighted':None,
        'graphic':{
            'mean_score':[],
            'mean_iou':[],
            'med_iou':[],
            'q25_iou':[],
            'q75_iou':[],
            'q10_iou':[],
            'q90_iou':[],
            'min_iou':[],
            'max_iou':[],
        },
    }

    for data_bin in prepared_data_bins:
        if len(data_bin[metric])>=bin_thr:
            res['graphic']['mean_score'].append(np.mean(data_bin['score']))
            res['graphic']['mean_iou'].append(np.mean(data_bin[metric]))
            res['graphic']['med_iou'].append(np.median(data_bin[metric]))
            res['graphic']['q25_iou'].append(np.percentile(data_bin[metric],25))
            res['graphic']['q75_iou'].append(np.percentile(data_bin[metric],75))
            res['graphic']['q10_iou'].append(np.percentile(data_bin[metric],10))
            res['graphic']['q90_iou'].append(np.percentile(data_bin[metric],90))
            res['graphic']['min_iou'].append(np.min(data_bin[metric]))
            res['graphic']['max_iou'].append(np.max(data_bin[metric]))

    for key in res['graphic']:
        res['graphic'][key] = np.array(res['graphic'][key])
    
    num_element = np.array([len(b['score']) for b in prepared_data_bins])

    num_element = num_element[num_element > bin_thr]
    MAEs = np.abs(res['graphic']['mean_score'] - res['graphic']['mean_iou'])
    
    res['ECE_regr'] = np.dot(MAEs,num_element.T)/np.sum(num_element)
    res['ECE_regr_unweighted'] = np.mean(MAEs)

    
    return res
    
    
def plot_iou_metrics(res, splits_lim=None, im_path=None):
    metric_key = res['metric']
    
    data = res['graphic']
    mean_score = data['mean_score']
    mean_iou = data['mean_iou']
    med_iou = data['med_iou']
    q25_iou = data['q25_iou']
    q75_iou = data['q75_iou']
    q10_iou = data['q10_iou']
    q90_iou = data['q90_iou']
    min_iou = data['min_iou']
    max_iou = data['max_iou']



    plt.figure(figsize=(10, 6))
    plt.scatter(mean_score, mean_iou, color='green', marker='o', label='Mean IoU')
    for i in range(len(mean_score)):
        #plt.plot([mean_score[i], mean_score[i]], [min_iou[i], max_iou[i]], color='blue', linestyle='-', linewidth=1)
        plt.plot([mean_score[i], mean_score[i]], [q25_iou[i], q75_iou[i]], color='blue', linestyle='-', linewidth=1)
        plt.plot(mean_score[i], q25_iou[i], color='blue', marker='_', markersize=3)
        plt.plot(mean_score[i], q75_iou[i], color='blue', marker='_', markersize=3)
        plt.plot(mean_score[i], med_iou[i], color='red', marker='_', markersize=5)

    
    ECE_regr= res['ECE_regr_unweighted']
    plt.title(f'Evolution of IoU Metrics for {metric_key} (ECE_regr: {ECE_regr})')
    plt.xlabel('Score')
    plt.ylabel(f'{metric_key}')
    plt.legend()
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if splits_lim is not None:
        for split in splits_lim:
            plt.axvline(x=split, color='gray', linestyle='-', linewidth=1)
    
    file_name = f"{metric_key}_ece_regr.png"
    save_path = os.path.join(im_path, file_name)
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")
    
    plt.show()




if __name__ == '__main__':
    args = parse_args()
    
    dt_path = Path(args.dt_path) / 'dt_data.pkl'

    with open(dt_path, 'rb') as dt_file:
        dt_data = pickle.load(dt_file)

    splits_lim = np.linspace(0, 1, 11)
    names = 'Any'
    
    prepared_data = prepare_data_ECE_regr(dt_data=dt_data, splits_lim=splits_lim, names=names)
    
    res_path = Path(args.dt_path) / 'ECE_regr.pkl'
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
        
    res=[]
    for m in ['iou_3d', 'iou_score_3d', 'iou_bev', 'iou_score_bev']:
        res_i = compute_ECE_regr(prepared_data, metric=m)
        res.append(res_i)
        plot_iou_metrics(res=res_i, im_path=im_path, splits_lim=splits_lim)
        
    with open(res_path, 'wb') as file:
        pickle.dump(res, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")
    

