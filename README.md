Ideas

1) Combine both types of images together (Did not work)
2) pseudo label
3) tta
4) Ensemble , try to see if any augs improve performance
5) change loss focal to see difference
6) See if big model improve Cv
7) `models = {
   'convnext_large_in22k': {
   (Resize(res), (320,224)),
   }, 'vit_large_patch16_224': {
   (Resize(480, method='squish'), 224),
   (Resize(res), 224),
   }, 'swinv2_large_window12_192_22k': {
   (Resize(480, method='squish'), 192),
   (Resize(res), 192),
   }, 'swin_large_patch4_window7_224': {
   (Resize(res), 224),
   }
   }`
8) Try to add EfficientNet And BEIT Model to this and maybe increase image size of the models if doable


Mapping
{'blast': 0, 'brown': 1, 'healthy': 2}