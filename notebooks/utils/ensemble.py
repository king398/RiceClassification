import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

id = \
    pd.read_csv(
        '/home/mithil/PycharmProjects/Rice/submission/convnext_base_384_in22ft1k_mixup_rgbonly_512_image_size.csv')[
        'filename'].values
probablity_1 = np.load(
    '/home/mithil/PycharmProjects/Rice/submission/swin_v2_base_25_epoch_no_mixup_tta.npy',
    allow_pickle=True)
probablity_2 = np.load(
    '/home/mithil/PycharmProjects/Rice/submission/swinv2_large_window12to24_192to384_22kft1k_mixup_25_epoch_tta.npy',
    allow_pickle=True)
probablity_3 = np.load(
    '/home/mithil/PycharmProjects/Rice/submission/swinv2_large_window12to24_192to384_22kft1k_tta.npy',
    allow_pickle=True)
probablity = probablity_1 * 0.40140872 + probablity_2 * 0.34685097 + probablity_3 * 0.25174032
blast = []
brown = []
healthy = []
probabilitys = probablity
for i in probabilitys:
    blast.append(i[0])
    brown.append(i[1])
    healthy.append(i[2])
lists = {0: 'blast', 1: 'brown', 2: 'healthy'}
probabilitys = torch.tensor(probablity)

probabilitys = list(np.array(torch.argmax(probabilitys, 1)))
other = pd.DataFrame({'labels': probabilitys})
other['labels'].replace(lists,inplace=True)
other['labels'].hist()
plt.show()
sub = pd.DataFrame({"filename": id, "blast": blast, "brown": brown, "healthy": healthy})
sub.to_csv(
    '/home/mithil/PycharmProjects/Rice/submission/ensemble/swin_v2_base_swin_v2_large_mixup_swinv2_large_window_12.csv',
    index=False)
