from .wrappers import AudioClassifier
import torch

class CNN1D(AudioClassifier):
    def __init__(self, optimizer, lr_scheduler=None, loss=None, metrics=None,
                 cnn_layer=torch.nn.Conv1d, 
                 channels=[64,128,128,256,256,512],
                 kernel_sizes=[16,16,8,8,4,4], strides=[4,4,2,2,2,2],
                 pooling_type='mean', classification_layer=torch.nn.Linear,
                 num_classes=None, key_in='wav'):

        super().__init__(optimizer, lr_scheduler, loss, metrics, num_classes)
        ch_ins = [1] + channels[:-1]
        ch_outs = channels

        self.encoder = torch.nn.Sequential(*[cnn_layer(ci,co,k,s) for ci,co,k,s in zip(ch_ins, ch_outs, kernel_sizes, strides)])
        self.classification_layer = classification_layer(channels[-1], num_classes)
        self.pooling_type = pooling_type
        self.key_in = 'wav'

    def forward(self, x):
        #x (BS, T)
        embeddings = self.encoder(x[self.key_in].unsqueeze(1)) # (BS, C, T)
        downsampling_factor = x['wav'].shape[-1]//embeddings.shape[-1]
        if 'wav_lens' in x:
            x['feature_lens'] = x['wav_lens']//downsampling_factor
            mask = torch.arange(embeddings.shape[-1], device=x['wav'].device)[None,None,:] < x['feature_lens'][:,None,None]
        else:
            mask = torch.ones_like(embeddings)
        if self.pooling_type == 'mean':
            embeddings = (embeddings*mask).sum(axis=-1)/mask.sum(axis=-1)
        x['yhat'] = self.classification_layer(embeddings)

    def predict(self, x):
        with torch.no_grad():
            self(x)