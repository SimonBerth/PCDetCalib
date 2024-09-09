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
    file_path = f"./results/{file_name}/spars.pkl"
    
    # Load the pickle file content
    with open(file_path, 'rb') as file:
        # Store the loaded data in a variable dynamically named after the model
        data[file_to_model[file_name]] = pickle.load(file)



plt.figure(figsize=(14, 6))

for key in data:
    dat = data[key]
    m_diff = dat['m_diff_norm']
    index = np.arange(len(m_diff))/len(m_diff)
    score = dat['score']

    int_diff = dat['int_diff_norm']
    amax_score = score[np.argmax(m_diff)]

    label = f"{key} - AUC:{int_diff : .2e} / Score max:{amax_score : .2f}"
    plt.plot(index, m_diff, label=label)
    
plt.title(f'Distance between oracle and sparsification curves')
plt.xlabel('% of detections eliminated')
plt.ylabel('Distance between curves')
plt.legend(loc='upper right', ncol=2)
plt.xlim(0, 1)
plt.ylim(0, 0.3)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)
plt.minorticks_on()
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
file_name = "spars_curb_norm.pdf"
save_path = os.path.join(Path('./ResPaper'), file_name)
plt.savefig(save_path)
plt.show()



plt.figure(figsize=(14, 6))

for key in data:
    dat = data[key]
    m_diff = dat['m_diff_norm']
    index = np.arange(len(m_diff))/len(m_diff)
    score = dat['score']

    amax_score = score[np.argmax(m_diff)]
    score = np.append(score, 1.0)
    int_diff = np.sum(m_diff * (score[1:] - score[:-1]))
    
    label = f"{key} - AUC:{int_diff : .2e} / Score max:{amax_score : .2f}"
    plt.plot(dat['score'], m_diff, label=label)
    

plt.title(f'Distance between oracle and sparsification curves')
plt.xlabel('Score associated to last removed BBox for sparsification curve')
plt.ylabel('Distance between curves')
plt.legend(loc='upper right', ncol=2)
plt.xlim(0, 1)
plt.ylim(0, 0.3)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)
plt.minorticks_on()
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
file_name = "spars_curb_norm_score.pdf"
save_path = os.path.join(Path('./ResPaper'), file_name)
plt.savefig(save_path)
plt.show()


plt.figure(figsize=(14, 6))

for key in data:
    dat = data[key]
    m_diff = dat['m_diff']
    index = np.arange(len(m_diff))/len(m_diff)
    score = dat['score']

    int_diff = dat['int_diff']
    amax_score = score[np.argmax(m_diff)]

    label = f"{key} - AUC:{int_diff : .2e} / Score max:{amax_score : .2f}"
    plt.plot(index, m_diff, label=label)
    
plt.title(f'Distance between oracle and sparsification curves')
plt.xlabel('% of detections eliminated')
plt.ylabel('Distance between curves')
plt.legend(loc='upper right', ncol=2)
plt.xlim(0, 1)
plt.ylim(0, 0.3)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)
plt.minorticks_on()
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
file_name = "spars_curb.pdf"
save_path = os.path.join(Path('./ResPaper'), file_name)
plt.savefig(save_path)
plt.show()



plt.figure(figsize=(14, 6))

for key in data:
    dat = data[key]
    m_diff = dat['m_diff']
    index = np.arange(len(m_diff))/len(m_diff)
    score = dat['score']

    amax_score = score[np.argmax(m_diff)]
    score = np.append(score, 1.0)
    int_diff = np.sum(m_diff * (score[1:] - score[:-1]))
    
    label = f"{key} - AUC:{int_diff : .2e} / Score max:{amax_score : .2f}"
    plt.plot(dat['score'], m_diff, label=label)
    


    
plt.title(f'Distance between oracle and sparsification curves')
plt.xlabel('Score associated to last removed BBox for sparsification curve')
plt.ylabel('Distance between curves')
plt.legend(loc='upper right', ncol=2)
plt.xlim(0, 1)
plt.ylim(0, 0.3)
plt.grid(True, which='major', linestyle='-', linewidth=0.8)
plt.minorticks_on()
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
file_name = "spars_curb_score.pdf"
save_path = os.path.join(Path('./ResPaper'), file_name)
plt.savefig(save_path)
plt.show()


fig, axs = plt.subplots(5, 2, figsize=(10, 10))
j=0
for key in data:
    ax = axs[j // 2, j % 2]
    j = j + 1
    
    dat = data[key]
    m_jaccard_score = dat['m_jaccard_score']
    m_jaccard_iou = dat['m_jaccard_iou']

    index = np.arange(len(m_diff))/len(m_diff)
    score = dat['score']
    
    ax.plot(index, m_jaccard_score, linestyle='-', color='green')
    ax.plot(index, m_jaccard_iou, linestyle='--', color='red')

    ax.set_title(key)
    #ax.set_xlabel('Mean score per bin')
    #ax.set_ylabel('Mean target value per bin')
    #plt.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))




plt.tight_layout(rect=[0.05,0.05,1,0.97])

fig.text(0.5,0.04, '% of eliminated detections', ha='center', fontsize=14)
fig.text(0.04,0.5, 'Oracl curve (red) and Sparsification curve (green)', va='center', rotation='vertical', fontsize=14)

file_name = "spars_complet.pdf"
save_path = os.path.join(Path('./ResPaper'), file_name)
fig.savefig(save_path)
fig.show()

plt.close()
