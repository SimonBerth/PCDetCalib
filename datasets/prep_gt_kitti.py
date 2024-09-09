import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.join(script_dir, '..', 'tools')
sys.path.append(tools_dir)
import argparse
from prep_data_kitti import get_kitti_info
from pathlib import Path
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description='Format the given results for Kitti val')
    parser.add_argument('kitti_path', nargs='?', default=script_dir + '/kitti', help='path of the folder containing training and ImageSets for Kitti')
    parser.add_argument(
        '--relative_path', type=bool, default=True,
        help='wether the given path is relative or absolute')
    args = parser.parse_args()
    return args

def split_to_ids(file_path):
    ids_list = []
    with open(file_path, 'r') as file:
        for line in file:
            number = int(line)
            ids_list.append(number)
    return ids_list

def get_gt_kitti(kitti_path, relative_path, clean_det=True):
    split_val_path = Path(kitti_path) / 'ImageSets/val.txt'
    ids=split_to_ids(split_val_path)
    
    gt_data = get_kitti_info(kitti_path, aux_info=True, ids=ids,
    num_worker=8, relative_path=relative_path, ground_truth=True, clean_det=clean_det)
    
    return gt_data


if __name__ == '__main__':
    args = parse_args()
    
    gt_data = get_gt_kitti(args.kitti_path, args.relative_path, clean_det=True)
    
    gt_path = Path(args.kitti_path) / 'gt_data.pkl'
    with open(gt_path, 'wb') as file:
        pickle.dump(gt_data, file)
    abs_gt_path = os.path.abspath(gt_path)
    print(f"Data has been saved to {abs_gt_path}")
    
