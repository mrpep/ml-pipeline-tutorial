import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import pandas as pd
from loguru import logger
import re
import copy
from torch.utils.data import Dataset, DataLoader
import inspect

def set_seed(state, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state.seed = seed
    return state

def load_dataset(state, reader_fn, 
                 cache=True, 
                 filters=[],
                 key_out='dataset_metadata'):
    
    if not (cache and key_out in state):
        if not isinstance(reader_fn, list):
            reader_fn = [reader_fn]
        dfs = [fn() for fn in reader_fn]
        df = pd.concat(dfs).reset_index()
        state[key_out] = df
    else:
        logger.info('Caching dataset metadata from state')
    
    for f in filters:
        state[key_out] = f(state[key_out])

    return state

def load_gtzan(data_dir):
    all_wavs = Path(data_dir).rglob('*.wav')
    metadata = []
    for w in tqdm(all_wavs):
        wav_info = sf.info(w)
        mi = {'filename': str(w.resolve()),
              'frames': wav_info.frames,
              'duration': wav_info.duration,
              'sample_id': w.stem.split('.')[1],
              'genre': w.stem.split('.')[0],
              'sr': wav_info.samplerate,
              'dataset': 'gtzan'}
        metadata.append(mi)
    return pd.DataFrame(metadata)

def partition_by_re(state, column_in, key_in='dataset_metadata', res=None, column_out='partition'):
    def match(x, res):
        for k,v in res.items():
            if re.match(v,x):
                return k

    state[key_in][column_out] = state[key_in][column_in].apply(lambda x: match(x, res))
    return state

def make_labels(state, column_in, key_in='dataset_metadata', key_out='class_map', column_out='classID'):
    class_map = {c: i for i,c in enumerate(sorted(state[key_in][column_in].unique()))}
    state[key_out] = class_map
    state[key_in][column_out] = state[key_in][column_in].apply(lambda x: class_map[x])
    return state

class DataFrameDataset(Dataset):
    def __init__(self, metadata, out_cols, preprocessors=None):
        self._metadata = metadata
        self._out_cols = out_cols
        self._preprocessors = [p() for p in preprocessors]

    def __getitem__(self, idx):
        row = copy.deepcopy(self._metadata.iloc[idx])
        for p in self._preprocessors:
            row = p(row)
        out = {k: row[k] for k in self._out_cols}
        return out

    def __len__(self):
        return len(self._metadata)

class ProcessorReadAudio:
    def __init__(self, max_duration=None,
                       key_in='filename',
                       key_out='wav',
                       key_duration='duration',
                       key_sr='sr'):
        
        self.max_duration = max_duration
        self.key_in, self.key_out, self.key_duration, self.key_sr = key_in, key_out, key_duration, key_sr

    def __call__(self, row):
        if self.max_duration is not None:
            dur = row[self.key_duration]
            if dur > self.max_duration:
                max_frames = int(self.max_duration*row[self.key_sr])
                frames = int(dur*row[self.key_sr])
                start = random.randint(0, frames-max_frames)
                stop = start + max_frames
        else:
            start=0
            stop=None

        x, fs = sf.read(row[self.key_in], start=start, stop=stop, dtype=np.float32)
        row[self.key_out] = x
        return row

def get_dataloaders(state, dataset_cls, dataloader_cls,
                    key_dataset='dataset_metadata',
                    column_partition='partition',
                    key_dataset_out='datasets',
                    key_dataloaders_out='dataloaders'):

    df_metadata = state[key_dataset]
    state[key_dataset_out] = {k: v(df_metadata.loc[df_metadata[column_partition]==k]) for k,v in dataset_cls.items()}
    state[key_dataloaders_out] = {k: v(state[key_dataset_out][k]) for k,v in dataloader_cls.items()}

    return state

def fit_model(state, model_cls=None, trainer_cls=None, 
              key_dataloaders='dataloaders', key_out = 'model',
              from_checkpoint=None, checkpoint_folder='checkpoints'):

    #Automatically pass number of classes to model:
    if 'num_classes' in inspect.signature(model_cls.__init__).parameters and 'class_map' in state:
        kwargs = {'num_classes': len(state['class_map'])}
    else:
        kwargs = {}

    model = model_cls(**kwargs)
    trainer = trainer_cls()
    trainer.checkpoint_callback.dirpath = trainer.checkpoint_callback.dirpath + '/{}'.format(checkpoint_folder)
    trainer.fit(model,
                state[key_dataloaders]['train'],
                state[key_dataloaders]['validation'],
                ckpt_path=from_checkpoint)
    
    state[key_out] = model
    return state
