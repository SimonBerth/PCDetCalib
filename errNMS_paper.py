import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from pathlib import Path


# List of model names and corresponding file paths
file_names = [
    "CasA-PV_nmsfree", "CasA-V_nmsfree", "CasA-T_nmsfree",
    "PartA2_free_nmsfree", "PartA2_nmsfree", "pdv_nmsfree",
    "pv_rcnn_nmsfree", "second_ct3d_3cat_nmsfree",
    "second_iou_nmsfree", "voxel_rcnn_3classes_nmsfree"
]

file_to_model = {
    "CasA-PV_nmsfree":'CasA-PV',
    "CasA-V_nmsfree":'CasA-V',
    "CasA-T_nmsfree":'CasA-T',
    "PartA2_free_nmsfree":'Part-A²-free',
    "PartA2_nmsfree":'Part-A²-anchor',
    "pdv_nmsfree":'PDV',
    "pv_rcnn_nmsfree":'PV-RCNN',
    "second_ct3d_3cat_nmsfree":'CT3D',
    "second_iou_nmsfree":'SECOND-IoU',
    "voxel_rcnn_3classes_nmsfree":'Voxel R-CNN',
}

# Dictionary to store data for each model
data = {}

# Loop through the model names and load the content of each pkl file
for file_name in file_names:
    file_path = f"./results/{file_name}/NMS_err_prct.pkl"
    
    # Load the pickle file content
    with open(file_path, 'rb') as file:
        # Store the loaded data in a variable dynamically named after the model
        data[file_to_model[file_name]] = pickle.load(file)

print("% of GT no longer detected due to bad score pre-NMS (as a % of all correctly detected GT pre-NMS) :")
for key in data:
    print(f"{key} : {data[key]}")
