# General
import os
import pandas as pd
import numpy as np
#from tqdm.notebook import tqdm
from tqdm import tqdm
import cv2
import warnings; warnings.simplefilter("ignore", UserWarning)

import matplotlib
matplotlib.use('Agg')  # Make matplotlib use an non-interactive backend
import matplotlib.pyplot as plt



# Metrics
from sklearn.metrics import roc_auc_score

# Machine Learning Utilities
from sklearn.model_selection import train_test_split 

# Deep Learning
import torch
#import torchvision
import timm
from timm import create_model

import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
#from torch.optim.lr_scheduler import CosineAnnealingLR
from fastai.vision.all import *
from fastai.vision.learner import _update_first_layer

# Image Augmentation
#import albumentations
#from albumentations.pytorch.transforms import ToTensorV2


# Device Type
# Device Optimization
if torch.cuda.is_available():
    device = 'gpu'  # torch.device('cuda')
else:
    device = 'cpu'  # torch.device('cpu')    
#print(f'Using device: {device}')
print(f'Using device: '+device)


set_seed(999,reproducible=True)  # Now, lets set a fixed random seed for repeatability.

# Meta Data
batch_size = 128  # batch size - range from 8 to 128+ depending on available VRAM on the GPU
epochs = 3  # epochs for training
model_arch = 'resnext50_32x4d'  # model architecture



# Training Data
df = pd.read_csv(os.path.join('labels','train_labels.csv'))

train_data_dir = os.path.join('/', 'data', 'train')
df['path'] = df['id'].apply(lambda x: train_data_dir+os.path.join('/')+f'{x[0]}/{x}.npy')

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=999)

# Dataset Class
class SETIDataset:
    def __init__(self, df, spatial=True, sixchan=True):
        self.df = df
        self.spatial = spatial  # Whether to use a spatial or channelized orrientation
        self.sixchan = sixchan  # Whether to use all six channels or just the three 'ON' channels
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index].target
        filename = self.df.iloc[index].path
        data = np.load(filename).astype(np.float32)
        if not self.sixchan: data = data[::2].astype(np.float32)
        if self.spatial:
            data = np.vstack(data).transpose((1, 0))
            data = cv2.resize(data, dsize=(256,256))     
            data_tensor = torch.tensor(data).float().unsqueeze(0)
        else:
            data = np.transpose(data, (1,2,0))
            data = cv2.resize(data, dsize=(256,256))     
            data = np.transpose(data, (2, 0, 1)).astype(np.float32)
            data_tensor = torch.tensor(data).float()

        return (data_tensor, torch.tensor(label))

# Create Datasets
train_ds = SETIDataset(train_df)
valid_ds = SETIDataset(valid_df)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=8)

dls = DataLoaders(train_dl, valid_dl)


# Training
def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")

def create_timm_model(arch:str, n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                     concat_pool=True, **kwargs):
    "Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library"
    body = create_timm_body(arch, pretrained, None, n_in)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children()))
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else: head = custom_head
    model = nn.Sequential(body, head)
    if init is not None: apply_init(model[1], init)
    return model

def timm_learner(dls, arch:str, loss_func=None, pretrained=True, cut=None, splitter=None,
                y_range=None, config=None, n_in=3, n_out=None, normalize=True, **kwargs):
    "Build a convnet style learner from `dls` and `arch` using the `timm` library"
    if config is None: config = {}
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = create_timm_model(arch, n_out, default_split, pretrained, n_in=n_in, y_range=y_range, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=default_split, **kwargs)
    if pretrained: learn.freeze()
    return learn

def roc_auc(preds,targ):
    try: return roc_auc_score(targ.cpu(),preds.squeeze().cpu())
    except: return 0.5

learn = timm_learner(dls,model_arch,pretrained=True,n_in=1,n_out=1,metrics=[roc_auc], opt_func=ranger, loss_func=BCEWithLogitsLossFlat()).to_fp16()

learn.lr_find(show_plot=True)
plt.savefig('lr_plot.png')

learn.fit_one_cycle(epochs, 0.1, cbs=[ReduceLROnPlateau()])

learn.recorder.plot_loss()
plt.savefig('loss_plot.png')

learn = learn.to_fp32()

# Save Model
learn.save(os.path.join('models', 'resnext50_32x4d-10epochs-128batch'))
learn = learn.load(os.path.join('models', 'resnext50_32x4d-10epochs-128batch'))

# Inference

# Test Data
test_df = pd.read_csv(os.path.join(os.path.join('labels','sample_submission.csv'))
test_dir = os.path.join('/', 'data', 'test')
test_df['path'] = test_df['id'].apply(lambda x: test_dir+os.path.join('/')+f'{x[0]}/{x}.npy')

# Test Dataset
test_ds = SETIDataset(test_df)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8, shuffle=False)

# Make predictions
preds = []
for xb, _ in tqdm(test_dl):
    if device == 'gpu':
        with torch.no_grad(): output = learn.model(xb.cuda())
        preds.append(torch.sigmoid(output.float()).squeeze().cpu())
    elif device == 'cpu':
        with torch.no_grad(): output = learn.model(xb.cpu())
        preds.append(torch.sigmoid(output.float()).squeeze().cpu())
    else:
        print('device not recognized')
preds = torch.cat(preds) 

# Write submissions file
sample_df = pd.read_csv(os.path.join('labels','sample_submission.csv'))
sample_df['target'] = preds
sample_df.to_csv(os.path.join('labels','resnext50_32x4d-10epochs-128batch.csv'), index=False)