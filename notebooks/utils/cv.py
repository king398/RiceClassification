import glob
import numpy as np
import statistics

paths = glob.glob(
    '/home/mithil/PycharmProjects/Rice/models/fix_csv_models/convnext_base_cutmix_512_image_size/*.pth')
cv = []
for i in paths:
    i = i.split(".pth")[0]
    i = i.split("_")[-1]
    cv.append(float(i))
print((sum(cv) / len(cv)))
