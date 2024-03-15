import torch

class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dims, 
                 activation=torch.nn.ReLU, 
                 final_activation=False,
                 pool_input=False):

        super().__init__()
        dim_in = [dim_in] + hidden_dims
        dim_out = hidden_dims + [dim_out]
        layers = [torch.nn.Sequential(torch.nn.Linear(di,do), activation()) 
                                           for di,do in zip(dim_in[:-1],dim_out[:-1])]
        if final_activation:
            layers.append(torch.nn.Sequential(torch.nn.Linear(dim_in[-1],dim_out[-1]), activation()))
        else:
            layers.append(torch.nn.Linear(dim_in[-1],dim_out[-1]))
        self.model = torch.nn.Sequential(*layers)
        self.pool_input = pool_input

    def forward(self, x):
        if self.pool_input:
            x = x.mean(dim=1)
        return self.model(x)

class Conv1DNormAct(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel, stride, 
                        activation=torch.nn.ReLU, 
                        normalization=torch.nn.BatchNorm1d):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Conv1d(ch_in, ch_out, kernel, stride), normalization(ch_out), activation())

    def forward(self, x):
        return self.model(x)