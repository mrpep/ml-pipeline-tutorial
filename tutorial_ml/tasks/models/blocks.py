import torch

class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dims, activation=torch.nn.ReLU):
        super().__init__()
        dim_in = [dim_in] + hidden_dims
        dim_out = hidden_dims + [dim_out]
        self.model = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(di,do), activation()) 
                                           for di,do in zip(dim_in,dim_out)])

    def forward(self, x):
        return self.model(x)

class Conv1DNormAct(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel, stride, 
                        activation=torch.nn.ReLU, 
                        normalization=torch.nn.BatchNorm1d):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Conv1d(ch_in, ch_out, kernel, stride), normalization(ch_out), activation())

    def forward(self, x):
        return self.model(x)