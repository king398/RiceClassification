import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing

train_df = pd.read_csv('/home/mithil/PycharmProjects/Rice/data/train.csv')
label_encoder = preprocessing.LabelEncoder()
train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
labels = train_df['Label']
labels = torch.tensor(labels)
probablity_1 = np.load('/home/mithil/PycharmProjects/Rice/oof/convnext_small_512_image_size_no_aug_oof_tta.npy',
                       allow_pickle=True)
probablity_2 = np.load('/home/mithil/PycharmProjects/Rice/oof/swinv2_base_window12to24_192to384_22kft1k_tta.npy',
                       allow_pickle=True)
best_loss = np.inf
best_weight = 0
for i in range(20):
    i = i / 20
    probablity = torch.tensor(probablity_1 * i + probablity_2 * (1 - i))
    loss = nn.NLLLoss()
    loss_item = abs(loss(probablity, labels).item())
    if loss_item < best_loss:
        best_weight = i
        best_loss = loss_item
print(best_loss)
print(best_weight)