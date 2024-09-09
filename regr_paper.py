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
data_regr = {}
data_MAE = {}

# Loop through the model names and load the content of each pkl file
for file_name in file_names:
    file_path = f"./results/{file_name}/ECE_regr.pkl"
    # Load the pickle file content
    with open(file_path, 'rb') as file:
        # Store the loaded data in a variable dynamically named after the model
        data_regr[file_to_model[file_name]] = pickle.load(file)
    file_path = f"./results/{file_name}/NMS_eval.pkl"
    # Load the pickle file content
    with open(file_path, 'rb') as file:
        # Store the loaded data in a variable dynamically named after the model
        data_MAE[file_to_model[file_name]] = pickle.load(file)


plt.figure(figsize=(14, 6))
plt.plot([0,1],[0,1],color='red')

for key in data_regr:
    dat = data_regr[key][1]
    mean_score = dat['graphic']['mean_score']
    mean_iou = dat['graphic']['mean_iou']
    MAE = data_MAE[key][1]['MAE']
    ABC = dat['ECE_regr_unweighted']

    label = f"{key} - ABC:{ABC: .2e} / MAE:{MAE : .2e}"
    plt.scatter(mean_score, mean_iou, label=label)

#plt.title(f'Regression accuracy')
plt.xlabel('Mean score per bin')
plt.ylabel('Mean target value per bin')
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

file_name = "regr.pdf"
save_path = os.path.join(Path('./ResPaper'), file_name)
plt.savefig(save_path)
plt.show()




fig, axs = plt.subplots(5, 2, figsize=(10, 10))
j=0
for key in data_regr:
    ax = axs[j // 2, j % 2]
    j = j + 1
    
    dat = data_regr[key][1]['graphic']
    mean_score = dat['mean_score']
    mean_iou = dat['mean_iou']
    med_iou = dat['med_iou']
    q25_iou = dat['q25_iou']
    q75_iou = dat['q75_iou']
    q10_iou = dat['q10_iou']
    q90_iou = dat['q90_iou']
    min_iou = dat['min_iou']
    max_iou = dat['max_iou']

    ax.scatter(mean_score, mean_iou, color='green', marker='o')#, label='Mean IoU  score')
    for i in range(len(mean_score)):
        #plt.plot([mean_score[i], mean_score[i]], [min_iou[i], max_iou[i]], color='blue', linestyle='-', linewidth=1)
        ax.plot([mean_score[i], mean_score[i]], [q25_iou[i], q75_iou[i]], color='blue', linestyle='-', linewidth=1)
        ax.plot(mean_score[i], q25_iou[i], color='blue', marker='_', markersize=3)
        ax.plot(mean_score[i], q75_iou[i], color='blue', marker='_', markersize=3)
        ax.plot(mean_score[i], med_iou[i], color='red', marker='_', markersize=5)


    ax.set_title(key)
    #ax.set_xlabel('Mean score per bin')
    #ax.set_ylabel('Mean target value per bin')
    #plt.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, which='major', linestyle='-', linewidth=0.8)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    for split in np.linspace(0, 1, 11):
        ax.axvline(x=split, color='gray', linestyle='-', linewidth=1)




plt.tight_layout(rect=[0.05,0.05,1,0.97])

fig.text(0.5,0.04, 'Predicted score : $\hat{S}_{IoU}$', ha='center', fontsize=14)
fig.text(0.04,0.5, 'Target value : $S_{IoU}$', va='center', rotation='vertical', fontsize=14)

file_name = "regr_complet.pdf"
save_path = os.path.join(Path('./ResPaper'), file_name)
fig.savefig(save_path)
fig.show()

plt.close()


fig, axs = plt.subplots(5, 2, figsize=(10, 10))
j = 0
for key in data_regr:
    ax = axs[j // 2, j % 2]
    j = j + 1

    dat = data_MAE[key][1]
    m_iou_metric = dat['m_iou_metric']
    m_score = dat['m_score']
    
    index = np.arange(len(m_score)) / len(m_score)

    ax.scatter(index, m_score, s=1, color='red')
    ax.plot(index, m_iou_metric, linestyle='--', color='green')

    ax.set_title(key)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, which='major', linestyle='-', linewidth=0.8)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

plt.tight_layout(rect=[0.05, 0.05, 1, 0.97])
fig.text(0.5, 0.04, 'Normalized index of detections', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'IoU score prediction: $\hat{S}_{IoU}$ (red) / IoU score target : ${S}_{IoU}$ (green)', va='center', rotation='vertical', fontsize=14)

file_name = "NMS.pdf"
save_path = os.path.join(Path('./ResPaper'), file_name)
fig.savefig(save_path)
fig.show()
