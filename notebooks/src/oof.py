import argparse
import glob
from pathlib import Path

import pandas as pd
############# Deep learning Stuff #################
import ttach as tta
import yaml
from sklearn import preprocessing
####### Function Created by me ###############
from dataset import *
from model import *
from train_func import *


def main(cfg):
    train_df = pd.read_csv(cfg['train_file_path'])

    train_df['file_path'] = train_df['Image_id'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()
    label_encoder = preprocessing.LabelEncoder()
    train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
    oof_preds = None
    oof_probablity = None
    oof_ids = []
    oof_targets = []
    for fold in range(5):

        if fold in cfg['folds']:
            best_model_name = None
            best_loss = np.inf

            train = train_df[train_df['fold'] != fold].reset_index(drop=True)

            valid = train_df[train_df['fold'] == fold].reset_index(drop=True)

            train_path = train['file_path']
            train_labels = train['Label']
            valid_path = valid['file_path']
            valid_labels = valid['Label']
            valid_dataset = Cultivar_data_oof(valid_path, cfg, valid_labels, ids=valid['Image_id'].values,
                                              transform=get_valid_transforms(cfg['image_size']))
            val_loader = DataLoader(
                valid_dataset, batch_size=cfg['batch_size'], shuffle=False,
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
            )
            model = BaseModel(cfg)
            path = glob.glob(f"{cfg['model_dir']}/{cfg['model']}_fold{fold}*.pth")
            model.load_state_dict(torch.load(path[0]))
            model = model.to(device)
            ids, target, preds, probablity, accuracy = oof_fn(val_loader, model, cfg)
            print(f"Fold: {fold} Accuracy: {accuracy}")
            oof_preds = np.concatenate([oof_preds, preds]) if oof_preds is not None else preds
            oof_probablity = np.concatenate([oof_probablity, probablity]) if oof_probablity is not None else probablity
            oof_ids.extend(ids)
            oof_targets.extend(target)

            del model
            del val_loader
            del valid_dataset
            del ids, target, preds, probablity, accuracy
            torch.cuda.empty_cache()
            gc.collect()
