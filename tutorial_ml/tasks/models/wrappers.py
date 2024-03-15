import torch
import pytorch_lightning as pl
import inspect

class AudioClassifier(pl.LightningModule):
    """
    Lightning Module for audio classification tasks.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer for training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Defaults to None.
        loss (torch.nn.Module, optional): Loss function. Defaults to torch.nn.CrossEntropyLoss.
        metrics (list of callable, optional): List of evaluation metrics functions. Defaults to None.
        num_classes (int, optional): Number of classes for classification. Defaults to None.
    """
    def __init__(self, optimizer, lr_scheduler=None,
                 loss=torch.nn.CrossEntropyLoss,
                 metrics=None, num_classes=None):

        super().__init__()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.classification_loss = loss()
        metrics_dict = {}
        train_metrics = []
        val_metrics = []
        for m in metrics:
            if 'num_classes' in inspect.signature(m.__init__).parameters:
                kwargs = {'num_classes': num_classes}
            else:
                kwargs = {}
            train_metrics.append(m(**kwargs))
            val_metrics.append(m(**kwargs))
        self.metrics = torch.nn.ModuleDict({'train_metrics': torch.nn.ModuleList(train_metrics),
                                            'val_metrics': torch.nn.ModuleList(val_metrics)})

    def training_step(self, batch, batch_idx):
        self(batch)
        yhat = batch['yhat']
        y = batch['classID']
        loss = self.classification_loss(yhat,y)
        self.log_metrics('train', yhat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        self.predict(batch)
        yhat = batch['yhat']
        y = batch['classID']
        loss = self.classification_loss(yhat,y)
        self.log_metrics('val', yhat, y)
        self.log('val_loss', loss, prog_bar=True)

    def log_metrics(self, stage, yhat, y):
        for m in self.metrics[stage+'_metrics']:
            m(yhat,y)
            self.log(stage+'_'+m.__class__.__name__,m)

    def configure_optimizers(self):
        opt = self.optimizer(self.trainer.model.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(opt) if self.lr_scheduler is not None else None
        else:
            lr_scheduler = None
        del self.optimizer
        del self.lr_scheduler
        opt_config = {'optimizer': opt}
        if lr_scheduler is not None:
            opt_config['lr_scheduler'] = {'scheduler': lr_scheduler,
                                          'interval': 'step',
                                          'frequency': 1}

        return opt_config

class UpstreamDownstream(AudioClassifier):
    """
    Combined Upstream-Downstream model for audio classification tasks.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer for training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Defaults to None.
        loss (torch.nn.Module, optional): Loss function. Defaults to None.
        metrics (list of callable, optional): List of evaluation metrics functions. Defaults to None.
        upstream (torch.nn.Module, optional): Upstream model. Defaults to None.
        downstream (torch.nn.Module, optional): Downstream model. Defaults to None.
        num_classes (int, optional): Number of classes for classification. Defaults to None.
    """
    def __init__(self, optimizer, lr_scheduler=None, loss=None, metrics=None,
                 upstream=None,
                 downstream=None,
                 num_classes=None):

        super().__init__(optimizer, lr_scheduler, loss, metrics, num_classes)
        self.upstream = upstream()
        self.downstream = downstream(self.upstream.embedding_dim, num_classes)

    def forward(self, x):
        self.upstream(x)
        x['yhat'] = self.downstream(x['embeddings'])

    def predict(self, x):
        with torch.no_grad():
            self.upstream(x)
            x['yhat'] = self.downstream(x['embeddings'])