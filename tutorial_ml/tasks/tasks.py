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
import torchaudio

class BatchDynamicPadding:
    def __call__(self, x):
        def not_discarded(x):
            if x is None:
                return False
            else:
                return not any([xi is None for xi in x.values()])

        def get_len(x):
            if x.ndim == 0:
                return 1
            else:
                return x.shape[0]

        def pad_to_len(x, max_len):
            if x.ndim == 0:
                return x
            else:
                pad_spec = ((0,max_len-x.shape[0]),) + ((0,0),)*(x.ndim - 1)
                return np.pad(x,pad_spec)

        def to_torch(x):
            if isinstance(x, torch.Tensor):
                return x
            else:
                if x.dtype in [np.float64, np.float32, np.float16, 
                            np.complex64, np.complex128, 
                            np.int64, np.int32, np.int16, np.int8,
                            np.uint8, bool]:

                    return torch.from_numpy(x)
                else:
                    return x
                
        x_ = x
        x = [xi for xi in x if not_discarded(xi)]

        batch = {k: [np.array(xi[k]) for xi in x] for k in x[0]}
        batch_lens = {k: [get_len(x) for x in batch[k]] for k in batch.keys()}
        batch_max_lens = {k: max(v) for k,v in batch_lens.items()}
        batch = {k: np.stack([pad_to_len(x, batch_max_lens[k]) for x in batch[k]]) for k in batch.keys()}
        batch_lens = {k+'_lens': np.array(v) for k,v in batch_lens.items()}
        batch.update(batch_lens)
        batch = {k: to_torch(v) for k,v in batch.items()}

        return batch

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

def get_dataloaders(state, dataset_cls, dataloader_cls,
                    key_dataset='dataset_metadata',
                    column_partition='partition',
                    key_dataset_out='datasets',
                    key_dataloaders_out='dataloaders'):

    df_metadata = state[key_dataset]
    state[key_dataset_out] = {k: v(df_metadata.loc[df_metadata[column_partition]==k]) for k,v in dataset_cls.items()}
    state[key_dataloaders_out] = {k: v(state[key_dataset_out][k]) for k,v in dataloader_cls.items()}

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

def make_labels(state, column_in, key_in='dataset_metadata', key_out='class_map', column_out='classID'):
    class_map = {c: i for i,c in enumerate(sorted(state[key_in][column_in].unique()))}
    state[key_out] = class_map
    state[key_in][column_out] = state[key_in][column_in].apply(lambda x: class_map[x])
    return state

def partition_by_re(state, column_in, key_in='dataset_metadata', res=None, column_out='partition'):
    def match(x, res):
        for k,v in res.items():
            if re.match(v,x):
                return k

    state[key_in][column_out] = state[key_in][column_in].apply(lambda x: match(x, res))
    return state

class ProcessorMelspectrogram:
    def __init__(self, key_in='wav', 
                       key_out='mel', 
                       frame_length=25, 
                       frame_shift=10, 
                       high_freq=0, 
                       htk_compat=False, 
                       low_freq=20,
                       num_mel_bins=23,
                       sample_frequency=16000,
                       window_type='povey',
                       dither=0.0,
                       use_energy=False, 
                       norm_stats=[0,1]):

        self.key_in = key_in
        self.key_out = key_out
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.high_freq = high_freq
        self.htk_compat = htk_compat
        self.low_freq = low_freq
        self.num_mel_bins = num_mel_bins
        self.sample_frequency = sample_frequency
        self.window_type = window_type
        self.dither = dither
        self.use_energy = use_energy
        self.norm_stats = norm_stats

    def __call__(self, row):
        if row[self.key_in] is None:
            mel = None #Training is robust to loading errors, if None arrives then dynamic_pad_batch filters it
        else:
            kwargs = dict(frame_length=self.frame_length,
                            frame_shift=self.frame_shift, high_freq=self.high_freq,
                            htk_compat=self.htk_compat, low_freq=self.low_freq, num_mel_bins=self.num_mel_bins,
                            sample_frequency=self.sample_frequency, window_type=self.window_type, use_energy=self.use_energy)
            mel = torchaudio.compliance.kaldi.fbank(torch.from_numpy(row[self.key_in]).unsqueeze(0), **kwargs).numpy()
            mel = (mel-self.norm_stats[0])/self.norm_stats[1]
        row[self.key_out] = mel
        return row

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

def set_seed(state, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state.seed = seed
    return state