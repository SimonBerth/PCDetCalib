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


def compute_spars(dt_data, names='Any', num_bins = 100):
    assert (dt_data[0]['score'] is not None), "Score values are required"
    assert (dt_data[0]['iou_3d'] is not None), "IoU precomputation is required"

    
    prepared_data = {
        'name' : [],
        'score' : [],
        'd_jaccard': [],
    }
    
    nbr_det = len(dt_data)
    prepared_data['name'] = np.concatenate([ dt['name']
        for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['score'] = np.concatenate([ np.atleast_1d(dt['score'])
        for dt in dt_data if dt['score'] is not None ], 0)
    prepared_data['d_jaccard'] = np.concatenate([ 1 - dt['iou_3d']
        for dt in dt_data if dt['score'] is not None ], 0)
    mask_tp = np.concatenate([ dt['mask_tp']
        for dt in dt_data if dt['score'] is not None ], 0)
    mask_fp = np.concatenate([ dt['mask_fp']
        for dt in dt_data if dt['score'] is not None ], 0)
        
    #Keep only det of class names
    if names != 'Any':
        if isinstance(names, str):
            names = [names]
        class_mask = np.isin(prepared_data['name'], names)
        for key in prepared_data:
            prepared_data[key] = prepared_data[key][class_mask]
    
    #Eliminate ignored det
    ign_mask = mask_tp | mask_fp
    for key in prepared_data:
        prepared_data[key] = prepared_data[key][ign_mask]
    
    prepared_data_score = {
        'name' : [],
        'score' : [],
        'd_jaccard': [],
    }
    prepared_data_jaccard = {
        'name' : [],
        'score' : [],
        'd_jaccard': [],
    }

    sort_score = np.argsort(prepared_data['score'])
    for key in prepared_data:
        prepared_data_score[key] = prepared_data[key][sort_score]

    sort_jaccard = np.argsort(prepared_data['d_jaccard'])[::-1]
    for key in prepared_data:
        prepared_data_jaccard[key] = prepared_data[key][sort_jaccard]
    
    num_elem = len(prepared_data['name'])
    q = num_elem // num_bins
    r = num_elem % num_bins
    bin_sizes = [q] * (num_bins-r) + [q+1] * r
    bin_index = [0]
    for size in bin_sizes[:-1]:
        bin_index.append(bin_index[-1] + size)
        
        
    '''
    prepared_data_bins_score = []
    prepared_data_bins_jaccard = []
    bin_start = 0
    for i in range(len(bin_sizes)):
        prepared_data_bins_score.append({
            'name': prepared_data_score['name'][bin_start:bin_start+bin_sizes[i]],
            'd_jaccard': prepared_data_score['d_jaccard'][bin_start:bin_start+bin_sizes[i]],
            'uncert_score': prepared_data_score['uncert_score'][bin_start:bin_start+bin_sizes[i]],
        })
        prepared_data_bins_jaccard.append({
            'name': prepared_data_jaccard['name'][bin_start:bin_start+bin_sizes[i]],
            'd_jaccard': prepared_data_jaccard['d_jaccard'][bin_start:bin_start+bin_sizes[i]],
            'uncert_score': prepared_data_jaccard['uncert_score'][bin_start:bin_start+bin_sizes[i]],
        })
        bin_start += bin_sizes[i]
    '''

    res = {
        'score':[],
        'm_jaccard_score':[],
        'm_jaccard_iou':[],
        'm_diff':[],
        'int_diff':None,
        'm_jaccard_score_norm':[],
        'm_jaccard_iou_norm':[],
        'm_diff_norm':[],
        'int_diff_norm':None,
    }

    res['score'] = np.array(
        [prepared_data_score['score'][i] for i in bin_index])
    
    res['m_jaccard_score'] = np.array(
        [np.mean(prepared_data_score['d_jaccard'][i:]) for i in bin_index])
    res['m_jaccard_iou'] = np.array(
        [np.mean(prepared_data_jaccard['d_jaccard'][i:]) for i in bin_index])
    res['m_diff'] = res['m_jaccard_score'] - res['m_jaccard_iou']
    res['int_diff'] = np.mean(res['m_diff'])
    
    ind_norm = 1/res['m_jaccard_score'][0]
    
    res['m_jaccard_score_norm'] = res['m_jaccard_score'] * ind_norm
    res['m_jaccard_iou_norm'] = res['m_jaccard_iou'] * ind_norm
    res['m_diff_norm'] = res['m_diff'] * ind_norm
    res['int_diff_norm'] = res['int_diff'] * ind_norm

    
    
    return res

    
    
def plot_spars(res, im_path=None):

    m_jaccard_score = res['m_jaccard_score_norm']
    m_jaccard_iou = res['m_jaccard_iou_norm']
    m_diff = res['m_diff_norm']
    int_diff = res['int_diff_norm']
    
    index = np.arange(len(m_diff))/len(m_diff)

    plt.figure(figsize=(10, 6))
    plt.plot(index, m_jaccard_score, label='Score based order', linestyle='-', color='red')
    plt.plot(index, m_jaccard_iou, label='Best (Jaccard index based) order', linestyle='--', color='green')
    plt.title(f'Sparsification curbs (area between curbs: {int_diff})')
    plt.xlabel('% of detections eliminated')
    plt.ylabel('mean Jaccard index over kept detections')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    file_name = f"spars_curbs.png"
    save_path = os.path.join(im_path, file_name)
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(index, m_diff, label='Evolution of distance with the best order', linestyle='-', color='red')
    plt.title(f'Sparsification distance curb (area under curb: {int_diff})')
    plt.xlabel('% of detections eliminated')
    plt.ylabel('Distance between Jaccard index over kept detections')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    file_name = f"spars_diff.png"
    save_path = os.path.join(im_path, file_name)
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")
    plt.show()
    
if __name__ == '__main__':
    args = parse_args()
    
    names = 'Any'
    num_bins = 100
    
    dt_path = Path(args.dt_path) / 'dt_data.pkl'
    with open(dt_path, 'rb') as dt_file:
        dt_data = pickle.load(dt_file)

    res_path = Path(args.dt_path) / 'spars.pkl'
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
    
    
    res = compute_spars(dt_data, names=names, num_bins = num_bins)
    
    plot_spars(res=res, im_path=im_path)
        
    with open(res_path, 'wb') as file:
        pickle.dump(res, file)
    abs_res_path = os.path.abspath(res_path)
    print(f"Result data has been saved to {abs_res_path}")
