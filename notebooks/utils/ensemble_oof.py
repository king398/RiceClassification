import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

train_df = pd.read_csv('/home/mithil/PycharmProjects/Rice/data/train.csv')
label_encoder = preprocessing.LabelEncoder()
train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
labels = train_df['Label']
labels = torch.tensor(labels)
probablity_1 = torch.tensor(
    np.load('/home/mithil/PycharmProjects/Rice/oof/swin_v2_base_25_epoch_no_mixup_tta.npy',
            allow_pickle=True))
probablity_2 = torch.tensor(
    np.load('/home/mithil/PycharmProjects/Rice/oof/swinv2_large_window12to24_192to384_22kft1k_mixup_25_epoch_tta.npy',
            allow_pickle=True))
probablity_3 = torch.tensor(
    np.load('/home/mithil/PycharmProjects/Rice/oof/swinv2_large_window12to24_192to384_22kft1k_tta.npy',
            allow_pickle=True))
probablity_4 = torch.tensor(
    np.load(
        '/home/mithil/PycharmProjects/Rice/oof/swinv2_base_window12to24_192to384_22kft1k_25_epoch_delayed_mixup_tta.npy',
        allow_pickle=True))

best_loss = np.inf
best_weight = 0
loss_list = []
for x in tqdm(range(10000)):

    i = np.random.random(4)
    i /= i.sum()
    probablity = torch.log(probablity_1 * i[0] + probablity_2 * i[1] + probablity_3 * i[2] + probablity_4 * i[3])
   # probablity = torch.clip(probablity, 0.025, 0.975)
    loss = nn.NLLLoss()
    loss_item = (loss(probablity, labels).item())
    if loss_item < best_loss:
        best_weight = i
        best_loss = loss_item
    loss_list.append(loss_item)
    break
print(best_loss)
print(best_weight)
loss_list.sort(reverse=True)
plt.plot(loss_list)
plt.show()
