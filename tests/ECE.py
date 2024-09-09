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


def prepare_data_ECE(dt_data, splits_lim = np.linspace(0, 1, 11), names='Any'):
    assert (dt_data[0]['score'] is not None), "Score values are required"
    assert (dt_data[0]['mask_tp'] is not None), "TP precomputation is required"
    assert (dt_data[0]['mask_fp'] is not None), "FP precomputation is required"
    prepared_data = {
        'idx': [],
        'group_idx': [],
        'score': [],
        'mask_tp' : [],
        'mask_fp' : [],
        'name' : [],
    }
    
    nbr_det = len(dt_data)
    prepared_data['idx'] = np.concatenate(
        [ [dt['idx']]*len(dt['group_idx']) for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['group_idx'] = np.concatenate(
        [ dt['group_idx'] for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['score'] = np.concatenate(
        [ np.atleast_1d(dt['score']) for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['mask_tp'] = np.concatenate(
        [ dt['mask_tp'] for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['mask_fp'] = np.concatenate(
        [ dt['mask_fp'] for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['name'] = np.concatenate(
        [ dt['name'] for dt in dt_data if dt['score'] is not None ], 0)
      
    sort_order = np.argsort(prepared_data['score'])
    for key in prepared_data:
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

    bin_inds = np.digitize(prepared_data['score'], splits_lim[1:], right=True)

    prepared_data_bins = []
    for i in range(len(splits_lim)-1):
        bin_msk = (bin_inds==i)
        prepared_data_bins.append({
            'idx': prepared_data['idx'][bin_msk],
            'group_idx': prepared_data['group_idx'][bin_msk],
            'score': prepared_data['score'][bin_msk],
            'mask_tp': prepared_data['mask_tp'][bin_msk],
            'mask_fp': prepared_data['mask_fp'][bin_msk],
            'name': prepared_data['name'][bin_msk],
        })
        
    return prepared_data_bins


def compute_ECE(prepared_data_bins, bin_thr=1):
    '''
    names : str, list[str], or 'Any' for all detection regardless of class
    bin_thr : min number of element for a bin to be used
    '''
    
    res = {
        'ECE':None,
        'graphic':{
            'mean_score':[],
            'accuracy':[],
        },
    }

    for data_bin in prepared_data_bins:
        if len(data_bin)>=bin_thr:
            res['graphic']['mean_score'].append(np.mean(data_bin['score']))
            res['graphic']['accuracy'].append(np.sum(data_bin['mask_tp'])/np.sum(data_bin['mask_tp']|data_bin['mask_fp']))

    
    for m in res['graphic']:
        res['graphic'][m] = np.array(res['graphic'][m])
    
    num_element = np.array([len(b['score']) for b in prepared_data_bins])
    num_element = num_element * (num_element > bin_thr)
    MAEs = np.abs(res['graphic']['mean_score'] - res['graphic']['accuracy'])
    res['ECE'] = np.dot(MAEs,num_element.T)/np.sum(num_element)
        
    return res
    
    
def plot_acc_metrics(res, splits_lim=None, im_path=None):

    data = res['graphic']
    mean_score = data['mean_score']
    accuracy = data['accuracy']

    plt.figure(figsize=(10, 6))
    plt.scatter(mean_score, accuracy, color='green', marker='o', label='Mean IoU')
    
    ECE = res['ECE']
    plt.title(f'Evolution of detector calibration (ECE: {ECE})')
    plt.xlabel('Score')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if splits_lim is not None:
        for split in splits_lim:
            plt.axvline(x=split, color='gray', linestyle='-', linewidth=1)
    
    file_name = "ECE.png"
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
    
    prepared_data = prepare_data_ECE(dt_data=dt_data, splits_lim=splits_lim, names=names)
    
    res = compute_ECE(prepared_data)
    
    res_path = Path(args.dt_path) / 'ECE.pkl'
    with open(res_path, 'wb') as file:
        pickle.dump(res, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")
    
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

    
    plot_acc_metrics(res=res, im_path=im_path, splits_lim=splits_lim)
