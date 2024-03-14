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
    def __init__(self, encodecmae_model,
                 layer=-1):

        super().__init__()
        self.encodecmae_model = load_model(encodecmae_model)
        self.embedding_dim = self.encodecmae_model.visible_encoder.model_dim
        self.embedding_rate = 75
        self.encodecmae_model.visible_encoder.compile=False

    def forward(self, x):
        from IPython import embed; embed()