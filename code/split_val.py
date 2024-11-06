import json
import numpy as np
import os.path as osp
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold

_lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
root_dir = 'data'
split = 'train'

y = []
groups = []

total_anno = dict(images=dict())
for label, nation in enumerate(_lang_list):
    with open(osp.join(root_dir, '{}_receipt/ufo/{}.json'.format(nation, split)), 'r', encoding='utf-8') as f:
        anno = json.load(f)
    for im in anno['images']:
        # print(im) # jpg name
        total_anno['images'][im] = anno['images'][im]
        y.append(label)
        groups.append(im)
print("Total:", len(total_anno['images']))
X = np.ones((len(total_anno['images']),1))
y = np.array(y)
groups = np.array(groups)

cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=411)

# check distribution
def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]

distrs = [get_distribution(y)]
print(distrs)
index = ['training set']

for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
    train_y, val_y = y[train_idx], y[val_idx]
    train_gr, val_gr = groups[train_idx], groups[val_idx]

    assert len(set(train_gr) & set(val_gr)) == 0 
    
    train_json = dict(images=dict())
    for im in train_gr:
        train_json['images'][im] = total_anno['images'][im]

    val_json = dict(images=dict())
    for im in val_gr:
        val_json['images'][im] = total_anno['images'][im]

    print(len(train_y)) # 300
    print(len(val_y)) # 100

    with open('train_split.json', 'w') as f:
        json.dump(train_json, f, indent=4)

    with open('val_split.json', 'w') as f:
        json.dump(val_json, f, indent=4)
    
    break # fold 하나만 생성
