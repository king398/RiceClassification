from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import pandas as pd

train_df = pd.read_csv('/home/mithil/PycharmProjects/Rice/data/train.csv')
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
label_encoder = preprocessing.LabelEncoder()
df = pd.DataFrame()
for fold, (trn_index, val_index) in enumerate(skf.split(train_df, train_df['Label'])):
    valid = train_df.iloc[val_index]
    valid = valid.reset_index(drop=True)
    valid['fold'] = fold
    df = pd.concat([df, valid]).reset_index(drop=True)
df.head()
del df['Unnamed: 0']
df.to_csv('/home/mithil/PycharmProjects/Rice/data/train.csv', index=False)
