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
    file_path = f"./results/{file_name}/ECE.pkl"
    
    # Load the pickle file content
    with open(file_path, 'rb') as file:
        # Store the loaded data in a variable dynamically named after the model
        data[file_to_model[file_name]] = pickle.load(file)

colors = [plt.cm.hsv(i / 10.0) for i in range(10)]  # 10 distinct colors in HSV

plt.figure(figsize=(14, 6))
i=0
for key in data:
    dat = data[key]
    mean_score = dat['graphic']['mean_score']
    accuracy = dat['graphic']['accuracy']
    ECE = dat['ECE']

    label = f"{key} - ECE:{ECE : .2e}"
    plt.scatter(mean_score,  accuracy,color=colors[i], label=label)
    i=i+1
    
#plt.title(f'Detector calibration')
plt.xlabel('Mean confidence per bin')
plt.ylabel('Accuracy per bin')
plt.legend(loc='upper left', ncol=2)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)
plt.minorticks_on()
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
for split in np.linspace(0, 1, 11):
    plt.axvline(x=split, color='gray', linestyle='-', linewidth=1)

file_name = "ECE.pdf"
save_path = os.path.join(Path('./ResPaper'), file_name)
plt.savefig(save_path)
plt.show()


for file_name in file_names:
    file_path = f"./results/{file_name}/dt_data.pkl"
    
    # Load the pickle file content
    with open(file_path, 'rb') as file:
        # Store the loaded data in a variable dynamically named after the model
        data[file_to_model[file_name]] = pickle.load(file)


plt.figure(figsize=(14, 6))

i=0
for key in data:
    dat = data[key]
    bins = np.arange(0, 1.1, 0.1)  # Bins from 0 to 1 in steps of 0.1
    scores = np.sort(np.concatenate([ np.atleast_1d(dt['score'])
        for dt in dat if dt['score'] is not None ], 0))
    counts, _ = np.histogram(scores, bins)
    counts = counts/np.sum(counts)
    
    label = f"{key}"
    plt.bar(bins[:-1], counts, width=0.1,linewidth=1.5, align='edge', edgecolor=colors[i], facecolor='none', label=label)
    
    i=i+1

plt.xlabel('Confidence')
plt.ylabel('% of det per bin')
plt.legend(loc='upper right', ncol=2)
plt.xlim(0, 1)
plt.ylim(0, 0.7)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)
plt.minorticks_on()
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
for split in np.linspace(0, 1, 11):
    plt.axvline(x=split, color='black', linestyle='-', linewidth=1.5)

file_name = "ECE_bins.pdf"
save_path = os.path.join(Path('./ResPaper'), file_name)
plt.savefig(save_path)
plt.show()
