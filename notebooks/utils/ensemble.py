import pandas as pd
import numpy as np

id = \
    pd.read_csv(
        '/home/mithil/PycharmProjects/Rice/submission/convnext_base_384_in22ft1k_mixup_rgbonly_512_image_size.csv')[
        'filename'].values
probablity_1 = np.load(
    '/home/mithil/PycharmProjects/Rice/submission/swin_large_patch4_window12_384_tta.npy',
    allow_pickle=True)
probablity_2 = np.load(
    '/home/mithil/PycharmProjects/Rice/submission/swinv2_base_window12to24_192to384_22kft1k_cutout_tta.npy',
    allow_pickle=True)
probablity_3 = np.load(
    '/home/mithil/PycharmProjects/Rice/submission/swinv2_large_window12to24_192to384_22kft1k_tta.npy',
    allow_pickle=True)

probablity = probablity_1 * 0.0564258 + probablity_2 * 0.50552098 + probablity_3 * 0.43805322
blast = []
brown = []
healthy = []
probabilitys = probablity
for i in probabilitys:
    blast.append(i[0])
    brown.append(i[1])
    healthy.append(i[2])
sub = pd.DataFrame({"filename": id, "blast": blast, "brown": brown, "healthy": healthy})
sub.to_csv(
    '/home/mithil/PycharmProjects/Rice/submission/ensemble/swin_v2_large_swin_v2_cutout_swin_v1_large.csv',
    index=False)
