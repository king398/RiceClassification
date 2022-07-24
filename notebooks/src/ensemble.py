import pandas as pd
import numpy as np

id = \
    pd.read_csv(
        '/home/mithil/PycharmProjects/Rice/submission/convnext_base_384_in22ft1k_mixup_rgbonly_512_image_size.csv')[
        'filename'].values
probablity_1 = np.load('/home/mithil/PycharmProjects/Rice/submission/convnext_small_512_image_size_no_aug_tta.npy',
                       allow_pickle=True)
probablity_2 = np.load('/home/mithil/PycharmProjects/Rice/submission/convnext_small_384_image_size.npy',
                       allow_pickle=True)
probablity = probablity_1 * 0.55 + probablity_2 * 0.45
blast = []
brown = []
healthy = []
probabilitys = probablity
for i in probabilitys:
    blast.append(i[0])
    brown.append(i[1])
    healthy.append(i[2])
sub = pd.DataFrame({"filename": id, "blast": blast, "brown": brown, "healthy": healthy})
sub.to_csv('/home/mithil/PycharmProjects/Rice/submission/ensemble/384_512_convnext_ensemble_2.csv', index=False)
