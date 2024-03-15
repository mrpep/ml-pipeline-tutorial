from .wrappers import AudioClassifier
import torch
import joblib
from pathlib import Path

def load_model(ckpt):
    state_path = Path(Path(ckpt).parent.parent,'state.pkl')
    state = joblib.load(state_path)
    model = state['model']
    model_ckpt = torch.load(ckpt)
    model.load_state_dict(model_ckpt['state_dict'])

    return model

class EnCodecMAEUpstream(torch.nn.Module):
    """
    Module to extract embeddings from an EnCodecMAE model.

    Args:
        encodecmae_model: Path to the EnCodecMAE model.
        layer (int, optional): Layer index to extract embeddings from. Defaults to -1.
    """
    def __init__(self, encodecmae_model,
                 layer=-1):

        super().__init__()
        self.encodecmae_model = load_model(encodecmae_model)
        self.embedding_dim = self.encodecmae_model.visible_encoder.model_dim
        self.embedding_rate = 75
        self.encodecmae_model.visible_encoder.compile=False
        self.encodecmae_model.train()
        self.layer = layer
        

    def forward(self, x):
        chunk_wav_size=96000
        chunk_feat_size=300
        wav_len = x['wav'].shape[1]
        feat_len = x['wav_features'].shape[1]
        acts = []
        for wi,fi in zip(range(0,wav_len,chunk_wav_size), range(0,feat_len,chunk_feat_size)):
            xi = {'wav': x['wav'][:,wi:wi+chunk_wav_size],
                  'wav_features': x['wav_features'][:, fi:fi+chunk_feat_size],
                  'wav_lens': torch.minimum(x['wav_lens'] - wi, torch.tensor(chunk_wav_size)),
                  'wav_features_lens': torch.minimum(x['wav_features_lens'] - fi, torch.tensor(chunk_feat_size))}
            self.encodecmae_model.wav_encoder(xi)
            self.encodecmae_model.masker.mask(xi,ignore_mask=True)
            self.encodecmae_model.visible_encoder(xi, return_activations=True)
            acts.append(xi['visible_embeddings_activations'][self.layer])
        acts = torch.cat(acts,dim=1)
        x['embeddings'] = acts


        