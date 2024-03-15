import torch

class MLP(torch.nn.Module):
    """
    Multi-layer perceptron (MLP) neural network model.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        hidden_dims (list of int): List of hidden layer dimensions.
        activation (torch.nn.Module, optional): Activation function for hidden layers. Defaults to torch.nn.ReLU.
        final_activation (bool, optional): Whether to apply activation function to the final output layer. Defaults to False.
        pool_input (bool, optional): Whether to pool the input data. Defaults to False.
    """

    def __init__(self, dim_in: int, dim_out: int, hidden_dims: list,
                 activation: torch.nn.Module = torch.nn.ReLU, 
                 final_activation: bool = False,
                 pool_input: bool = False):

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
    """
    1D convolutional layer with normalization and activation.

    Args:
        ch_in (int): Number of input channels.
        ch_out (int): Number of output channels.
        kernel (int): Convolutional kernel size.
        stride (int): Stride of the convolution.
        activation (torch.nn.Module, optional): Activation function. Defaults to torch.nn.ReLU.
        normalization (torch.nn.Module, optional): Normalization layer. Defaults to torch.nn.BatchNorm1d.
    """

    def __init__(self, ch_in: int, ch_out: int, kernel: int, stride: int, 
                 activation: torch.nn.Module = torch.nn.ReLU, 
                 normalization: torch.nn.Module = torch.nn.BatchNorm1d):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Conv1d(ch_in, ch_out, kernel, stride), normalization(ch_out), activation())

    def forward(self, x):
        return self.model(x)