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
from scipy.special import softmax
from typing import List, Dict, Optional, Callable, Any, Union

class BatchDynamicPadding:
    """
    Class for dynamically padding batches of variable-length sequences and converting them to PyTorch tensors.
    """
    def __call__(self, x):
        """
        Performs dynamic padding on a batch of variable-length sequences and converts them to PyTorch tensors.

        Args:
            x (list): List of dictionaries, where each dictionary represents a sample with keys corresponding to different features and values being sequences of variable lengths.

        Returns:
            dict: Dictionary containing padded sequences converted to PyTorch tensors. Keys represent feature names, and values are corresponding tensors. Additional keys are added for the lengths of sequences in each batch.
        """
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
    """
    Dataset class for handling data stored in Pandas DataFrame.

    Args:
        metadata (pandas.DataFrame): DataFrame containing the dataset.
        out_cols (list): List of column names in the DataFrame to be used as output.
        preprocessors (list, optional): List of preprocessor functions to apply on each row of the DataFrame. Defaults to None.
    """
    def __init__(self, metadata: pd.DataFrame, out_cols: List[str], preprocessors: Optional[List[Callable]] = None):
        self._metadata = metadata
        self._out_cols = out_cols
        self._preprocessors = [p() for p in preprocessors]

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing data for the specified index.
        """
        row = copy.deepcopy(self._metadata.iloc[idx])
        for p in self._preprocessors:
            row = p(row)
        out = {k: row[k] for k in self._out_cols}
        return out

    def __len__(self):
        return len(self._metadata)

def discard_samples(df: pd.DataFrame, column: str, list_file: str) -> pd.DataFrame:
    """
    Discards samples from a DataFrame based on values in a specified column.

    Args:
        df (pandas.DataFrame): DataFrame containing the samples.
        column (str): Name of the column containing values to filter.
        list_file (str): Path to the file containing values to discard.

    Returns:
        pandas.DataFrame: DataFrame with samples filtered out based on the values in the specified column.
    """
    with open(list_file,'r') as f:
        discard = f.read().splitlines()
    df = df.loc[~df[column].isin(discard)]
    return df

def find_last_ckpt(path):
    ckpts = list(Path(path).rglob('*.ckpt'))
    if len(ckpts) > 0:
        stems = [p.stem for p in ckpts]
        print(stems)
        if 'last' in stems:
            ckpt = Path(path, 'last.ckpt')
        else:
            steps = [int(p.stem.split('-')[-1].split('step=')[-1]) for p in ckpts]
            ckpt = ckpts[steps.index(max(steps))]
        logger.info('Restoring training from checkpoint: {}'.format(ckpt))
    else:
        ckpt = None
    return ckpt

def fit_model(state: Dict[str, Any], model_cls=None, trainer_cls=None,
              key_dataloaders: str = 'dataloaders', key_out: str = 'model',
              from_checkpoint: str = 'last', checkpoint_folder: str = 'checkpoints') -> Dict[str, Any]:
    """
    Fits a model using the provided training and validation data loaders and saves the best model checkpoint.

    Args:
        state (dict): Dictionary containing the state of the training process.
        model_cls (class, optional): Model class to instantiate. Defaults to None.
        trainer_cls (class, optional): Trainer class to instantiate. Defaults to None.
        key_dataloaders (str, optional): Key in the state dictionary for the data loaders. Defaults to 'dataloaders'.
        key_out (str, optional): Key in the state dictionary to store the trained model. Defaults to 'model'.
        from_checkpoint (str, optional): Specifies whether to resume training from a checkpoint. Defaults to 'last'.
        checkpoint_folder (str, optional): Folder name for saving model checkpoints. Defaults to 'checkpoints'.

    Returns:
        dict: Updated state dictionary with the trained model and the path to the best model checkpoint.
    """

    #Automatically pass number of classes to model:
    if 'num_classes' in inspect.signature(model_cls.__init__).parameters and 'class_map' in state:
        kwargs = {'num_classes': len(state['class_map'])}
    else:
        kwargs = {}

    model = model_cls(**kwargs)
    trainer = trainer_cls()
    trainer.checkpoint_callback.dirpath = trainer.checkpoint_callback.dirpath + '/{}'.format(checkpoint_folder)

    if from_checkpoint == 'last':
        from_checkpoint = find_last_ckpt(trainer.checkpoint_callback.dirpath)

    trainer.fit(model,
                state[key_dataloaders]['train'],
                state[key_dataloaders]['validation'],
                ckpt_path=from_checkpoint)

    state[key_out] = model
    state['best_model_path'] = trainer.checkpoint_callback.best_model_path

    return state

def eval_model(state: Dict[str, Any], key_model: str = 'model', key_dataloaders: str = 'dataloaders',
               metrics: Optional[List[Callable]] = None) -> Dict[str, Any]:
    """
    Evaluates a model on the test dataset using specified evaluation metrics.

    Args:
        state (dict): Dictionary containing the state of the evaluation process.
        key_model (str, optional): Key in the state dictionary for the trained model. Defaults to 'model'.
        key_dataloaders (str, optional): Key in the state dictionary for the data loaders. Defaults to 'dataloaders'.
        metrics (list of callables, optional): List of evaluation metrics functions. Defaults to None.

    Returns:
        dict: Updated state dictionary with evaluation metrics results.
    """
    model = state['model']
    model.load_state_dict(torch.load(state['best_model_path'])['state_dict'])
    model.eval()
    test_dataset = state[key_dataloaders]['test']
    logits = []
    targets = []
    for x in tqdm(test_dataset):
        model.predict(x)
        logits.append(x['yhat'].detach().cpu().numpy())
        targets.append(x['classID'].detach().cpu().numpy())
    targets = np.concatenate(targets)
    logits = np.concatenate(logits)

    metric_results = {}
    probs = softmax(logits,axis=-1)
    preds = np.argmax(probs,axis=-1)
    for m in metrics:
        type_yhat = list(inspect.signature(m).parameters.keys())[1]
        if 'score' in type_yhat:
            metric_results['test_{}'.format(m.__name__)] = m(targets, probs)
        elif 'pred' in type_yhat:
            metric_results['test_{}'.format(m.__name__)] = m(targets, preds)
        else:
            raise Exception('Second arg of metric should contain pred or score in its name')
    
    state['metrics'] = metric_results
    return state

def get_dataloaders(state: Dict[str, any], dataset_cls: Dict[str, any], dataloader_cls: Dict[str, any],
                    key_dataset: str = 'dataset_metadata', column_partition: str = 'partition',
                    key_dataset_out: str = 'datasets', key_dataloaders_out: str = 'dataloaders') -> Dict[str, any]:
    """
    Generates data loaders from dataset metadata using specified dataset and data loader classes.

    Args:
        state (dict): Dictionary containing the state of the data loading process.
        dataset_cls (dict): Dictionary mapping dataset class names to their respective classes.
        dataloader_cls (dict): Dictionary mapping data loader class names to their respective classes.
        key_dataset (str, optional): Key in the state dictionary for the dataset metadata. Defaults to 'dataset_metadata'.
        column_partition (str, optional): Column name in the dataset metadata indicating the partition. Defaults to 'partition'.
        key_dataset_out (str, optional): Key in the state dictionary to store the generated datasets. Defaults to 'datasets'.
        key_dataloaders_out (str, optional): Key in the state dictionary to store the generated data loaders. Defaults to 'dataloaders'.

    Returns:
        dict: Updated state dictionary with the generated datasets and data loaders.
    """

    df_metadata = state[key_dataset]
    state[key_dataset_out] = {k: v(df_metadata.loc[df_metadata[column_partition]==k]) for k,v in dataset_cls.items()}
    state[key_dataloaders_out] = {k: v(state[key_dataset_out][k]) for k,v in dataloader_cls.items()}

    return state

def load_dataset(state: dict, reader_fn: Union[Callable, List[Callable]], 
                 cache: bool = True, 
                 filters: Optional[List[Callable]] = None,
                 key_out: str = 'dataset_metadata') -> dict:
    """
    Loads dataset metadata using specified reader function(s) and applies optional filters.

    Args:
        state (dict): Dictionary containing the state of the dataset loading process.
        reader_fn (callable or list of callables): Reader function(s) to load dataset metadata.
        cache (bool, optional): Flag to cache dataset metadata in the state. Defaults to True.
        filters (list of callables, optional): List of filter functions to apply on the dataset metadata. Defaults to None.
        key_out (str, optional): Key in the state dictionary to store the loaded dataset metadata. Defaults to 'dataset_metadata'.

    Returns:
        dict: Updated state dictionary with the loaded dataset metadata.
    """
    
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

def load_gtzan(data_dir: str) -> pd.DataFrame:
    """
    Loads metadata for GTZAN dataset.

    Args:
        data_dir (str): Directory containing GTZAN dataset.

    Returns:
        pandas.DataFrame: DataFrame containing metadata for GTZAN dataset.
    """
    all_wavs = Path(data_dir).rglob('*.wav')
    metadata = []
    for w in tqdm(all_wavs):
        wav_info = sf.info(w)
        mi = {'filename': str(w.resolve()),
              'frames': wav_info.frames,
              'duration': wav_info.duration,
              'segment_id': w.stem.split('.')[1],
              'genre': w.stem.split('.')[0],
              'sr': wav_info.samplerate,
              'dataset': 'gtzan',
              'id':w.stem}
        metadata.append(mi)
    return pd.DataFrame(metadata)

def make_labels(state: dict, column_in: str, key_in: str = 'dataset_metadata', 
                key_out: str = 'class_map', column_out: str = 'classID') -> dict:
    """
    Creates a mapping of class labels to numeric IDs and applies the labels to the dataset.

    Args:
        state (dict): Dictionary containing the state of the labeling process.
        column_in (str): Name of the column containing class labels in the dataset.
        key_in (str, optional): Key in the state dictionary for the input dataset metadata. Defaults to 'dataset_metadata'.
        key_out (str, optional): Key in the state dictionary to store the generated class label mapping. Defaults to 'class_map'.
        column_out (str, optional): Name of the column to store the numeric class IDs in the dataset. Defaults to 'classID'.

    Returns:
        dict: Updated state dictionary with the class label mapping and dataset.
    """

    class_map = {c: i for i,c in enumerate(sorted(state[key_in][column_in].unique()))}
    state[key_out] = class_map
    state[key_in][column_out] = state[key_in][column_in].apply(lambda x: class_map[x])
    return state

def partition_by_re(state: dict, column_in: str, key_in: str = 'dataset_metadata', res: Dict[str, str] = None, 
                    column_out: str = 'partition') -> dict:
    """
    Partitions the dataset based on regular expressions.

    Args:
        state (dict): Dictionary containing the state of the partitioning process.
        column_in (str): Name of the column containing data to partition.
        key_in (str, optional): Key in the state dictionary for the input dataset metadata. Defaults to 'dataset_metadata'.
        res (dict, optional): Dictionary mapping partition names to regular expressions. Defaults to None.
        column_out (str, optional): Name of the column to store the partition labels in the dataset. Defaults to 'partition'.

    Returns:
        dict: Updated state dictionary with the partitioned dataset.
    """
    def match(x, res):
        for k,v in res.items():
            if re.match(v,x):
                return k

    state[key_in][column_out] = state[key_in][column_in].apply(lambda x: match(x, res))
    return state

class ProcessorMelspectrogram:
    """
    Processor class to compute Mel spectrograms from waveforms.

    Args:
        key_in (str, optional): Key in the input row for the waveform data. Defaults to 'wav'.
        key_out (str, optional): Key in the output row for the Mel spectrogram data. Defaults to 'mel'.
        frame_length (int, optional): Length of the analysis frame in milliseconds. Defaults to 25.
        frame_shift (int, optional): Shift between consecutive frames in milliseconds. Defaults to 10.
        high_freq (int, optional): High cutoff frequency for the Mel filter bank. Defaults to 0.
        htk_compat (bool, optional): Whether to use HTK compatibility. Defaults to False.
        low_freq (int, optional): Low cutoff frequency for the Mel filter bank. Defaults to 20.
        num_mel_bins (int, optional): Number of Mel bins in the filter bank. Defaults to 23.
        sample_frequency (int, optional): Sampling frequency of the input waveform. Defaults to 16000.
        window_type (str, optional): Type of window to use for spectrogram computation. Defaults to 'povey'.
        dither (float, optional): Dithering constant to add to the waveform. Defaults to 0.0.
        use_energy (bool, optional): Whether to use energy as the 0-th coefficient. Defaults to False.
        norm_stats (list, optional): Normalization statistics (mean and standard deviation) for the Mel spectrogram. Defaults to [0, 1].
    """
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
    """
    Processor class to read audio files and extract waveforms.

    Args:
        max_duration (float, optional): Maximum duration of audio to read (in seconds). Defaults to None.
        key_in (str, optional): Key in the input row for the audio file path. Defaults to 'filename'.
        key_out (str, optional): Key in the output row for the waveform data. Defaults to 'wav'.
        key_duration (str, optional): Key in the input row for the audio duration. Defaults to 'duration'.
        key_sr (str, optional): Key in the input row for the audio sample rate. Defaults to 'sr'.
    """
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
    """
    Sets the random seed for reproducibility.

    Args:
        state (dict): Dictionary containing the state of the seed setting process.
        seed (int, optional): Random seed value. Defaults to 42.

    Returns:
        dict: Updated state dictionary with the seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state.seed = seed
    return state

