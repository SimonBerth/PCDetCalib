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
    parser.add_argument('dt_path', help='path of the folder containing pred_instances_3d')
    parser.add_argument( '--relative_path', type=bool, default=True,
        help='wether the given path is relative or absolute')
    parser.add_argument( '--use_index', type=bool, default=True,
        help='recommended True if OpenPCDet, False if mmdetection3d')
    args = parser.parse_args()
    return args

def split_to_ids(file_path):
    ids_list = []
    with open(file_path, 'r') as file:
        for line in file:
            number = int(line)
            ids_list.append(number)
    return ids_list

def get_dt_kitti(dt_path,relative_path, use_index):
    if use_index:
        split_val_path = Path(script_dir) / 'ImageSets/val.txt'
        ids = split_to_ids(split_val_path)
    else:
        ids=3769

    dt_data = get_kitti_info(dt_path, aux_info=False, ids=ids,
    num_worker=8, relative_path=relative_path, ground_truth=False)
    
    return dt_data

if __name__ == '__main__':
    args = parse_args()
    
    dt_data = get_dt_kitti(args.dt_path, args.relative_path, use_index=args.use_index)

    dt_path = Path(args.dt_path) / 'dt_data.pkl'
    with open(dt_path, 'wb') as file:
        pickle.dump(dt_data, file)
    abs_dt_path = os.path.abspath(dt_path)
    print(f"Result data has been saved to {abs_dt_path}")
